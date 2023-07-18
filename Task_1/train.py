#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import argparse
import os
import signal
from argparse import ArgumentParser
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.logging import logger
from onmt.train_single import main as single_main


def main(opt):
    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if opt.epochs:
        raise AssertionError("-epochs is deprecated please use -train_steps.")

    if opt.truncated_decoder > 0 and opt.accum_count > 1:
        raise AssertionError("BPTT is not compatible with -accum > 1")

    if len(opt.gpuid) > 1:
        raise AssertionError("gpuid is deprecated \
              see world_size and gpu_ranks")
    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, ), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)

def add_training_args(parser, data_path, checkpoint_path=None):
    parser.add_argument('-data', type=str, required=data_path)
    parser.add_argument('-save_model', type=str, required=True)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-gpu_ranks', type=int, default=0)
    parser.add_argument('-save_checkpoint_steps', type=int, default=10000)
    parser.add_argument('-keep_checkpoint', type=int, default=20)
    parser.add_argument('-train_steps', type=int, default=500000)
    parser.add_argument('-param_init', type=float, default=0)
    parser.add_argument('-param_init_glorot', action='store_true')
    parser.add_argument('-max_generator_batches', type=int, default=32)
    parser.add_argument('-batch_size', type=int, default=4096)
    parser.add_argument('-batch_type', type=str, default='tokens')
    parser.add_argument('-normalization', type=str, default='tokens')
    parser.add_argument('-max_grad_norm', type=float, default=0)
    parser.add_argument('-accum_count', type=int, default=4)
    parser.add_argument('-optim', type=str, default='adam')
    parser.add_argument('-adam_beta1', type=float, default=0.9)
    parser.add_argument('-adam_beta2', type=float, default=0.998)
    parser.add_argument('-decay_method', type=str, default='noam')
    parser.add_argument('-warmup_steps', type=int, default=8000)
    parser.add_argument('-learning_rate', type=float, default=2)
    parser.add_argument('-label_smoothing', type=float, default=0.0)
    parser.add_argument('-report_every', type=int, default=1000)
    parser.add_argument('-layers', type=int, default=4)
    parser.add_argument('-rnn_size', type=int, default=256)
    parser.add_argument('-word_vec_size', type=int, default=256)
    parser.add_argument('-encoder_type', type=str, default='transformer')
    parser.add_argument('-decoder_type', type=str, default='transformer')
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-position_encoding', action='store_true')
    parser.add_argument('-share_embeddings', action='store_true')
    parser.add_argument('-global_attention', type=str, default='general')
    parser.add_argument('-global_attention_function', type=str, default='softmax')
    parser.add_argument('-self_attn_type', type=str, default='scaled-dot')
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-transformer_ff', type=int, default=2048)
    if checkpoint_path is not None:
        parser.add_argument('-train_from', type=str, default=checkpoint_path)
    return parser
