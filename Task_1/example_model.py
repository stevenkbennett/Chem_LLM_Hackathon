# Outlines the model architecture.
from onmt.translate.translator import build_translator
from argparse import ArgumentParser
from onmt.opts import translate_opts, model_opts, train_opts
import tempfile
from train import add_training_args
from train import main as trainer
from pathlib import Path

class Model:
    def __init__(self, model_path, gpu=-1):
        """Initializes the model.
        
        Args:
            model_path (str): The path to the model..
        """
        # Process arguments as a namedtuple for ONMT parsing.
        # Change to a list to allow for multiple models.
        if not isinstance(model_path, list):
            self._model_path = [str(Path(model_path).resolve())]
        parser = ArgumentParser()
        translate_opts(parser)
        output_path = tempfile.NamedTemporaryFile(delete=False).name
        # Perform argument parsing for ONMT.
        args = parser.parse_args(
            ['-model'] + self._model_path \
            + ['-output'] + [output_path] \
            + ['-src'] + ['None'] \
            + ['-gpu'] + [str(gpu)] \
        )
        self._model = build_translator(args, report_score=True)
        self._model.fast = True
        self._output_path = output_path
        # self._source_path = source_path
        self._attn_debug = False

    def predict(self, source_path, num_predictions=1, batch_size=1, beam_size=1):
        """Performs the prediction and returns the results.

        Args:
            num_predictions (int): The number of predictions to return per molecule.
            source_path (str): The path to the source file.
            batch_size (int): The batch size to use for prediction.
            beam_size (int): The beam size to use for prediction.

        """
        # Performs the prediction and writes to an output file
        res = {}
        # Number of predictions and beam size must be the same
        self._model.n_best = num_predictions
        self._model.beam_size = beam_size
        if beam_size < num_predictions:
            raise ValueError("Beam size must be greater than \
                             or equal to the number of predictions."
                            )
        # Read the source file
        with open(source_path, 'r') as f:
            source_mols = f.read().replace(" ", "").split('\n')

        self._model.translate(
            src_path=source_path,
            tgt_path=None,
            batch_size=batch_size,
            attn_debug=self._attn_debug,
        )
        with open(self._output_path, 'r') as f:
            preds = f.read().replace(" ", "").split('\n')[:-1]
        for i in range(len(source_mols)):
            res[source_mols[i]] = preds[i*num_predictions:(i+1)*num_predictions]
        return res
    
    def train(self, data_path, num_epochs=1, gpu_ranks=0):
        """Trains the model."""
        print(self._model_path)
        parser = ArgumentParser()
        model_opts(parser)
        train_opts(parser)
        opt = parser.parse_args(
            ['-data'] + [data_path] \
            + ['-train_from'] + [self._model_path[0]] \
            + ['-train_steps'] + [str(num_epochs)] \
            + ['-gpu_ranks'] + [str(gpu_ranks)]
        )
        trainer(opt)