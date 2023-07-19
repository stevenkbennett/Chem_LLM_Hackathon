# This script is used to evaluate the model on the test set, and is used to generate results. 
# The Evaluator class can be used to evalate the model on the test set, and generate results.

from rdkit import Chem
import numpy as np
import sys
from pathlib import Path
# import os
# import torch
# from transformers import BertModel
# from rxnfp.transformer_fingerprints import (
#     RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
# )
import pickle

import json

class Evaluator:
    """ABC for performing evaluations."""
    def __init__(self):
        """Initializes the evaluator.
        """
        ...
    
    def __call__(self, results: dict) -> float:
        """Boilerplate call function for performing evaluations."""
        raise NotImplementedError

    @staticmethod
    def is_valid_smiles(smi):
        """ Checks if a SMILES string is valid.
        """
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, kekuleSmiles=True)
            return True
        except Exception as e:
            return False

    def compare_smiles(self, smi1, smi2):
        """ Canonicalize SMILES and compare
        """
        # First, check if the SMILES are valid
        if self.is_valid_smiles(smi1) is False or self.is_valid_smiles(smi2) is False:
            return False
        smi1, smi2 = [
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) for smi in [smi1, smi2]
        ]
        return smi1 == smi2


class InvalidSMILES(Evaluator):
    def __init__(self):
        """Initializes the evaluator.
        """
        super().__init__()

    def __call__(self, results: dict) -> float:
        """Calculates the percentage of invalid reactant predictions"""
        tot_invalid = []
        for reaction in results.values():
            num_invalid = 0
            for reactants in reaction:
                if any(self.is_valid_smiles(smi) is False for smi in reactants.split(".")):
                    num_invalid += 1
                    continue
            tot_invalid.append(num_invalid / len(reaction) )

        return 1-np.mean(tot_invalid)


class TopK(Evaluator):
    """Performs a top-N evaluation."""
    def __init__(self, k, test_path=None):
        """Initializes the evaluator.
        
        Args:
            k (int): The number of predictions to consider.
        """
        self._k = k
        self._test_path = test_path
        if self._test_path is None:
            self._test_path = "test_task_1.txt"
        super().__init__()

    def __call__(self, results: dict) -> float:
        """Performs a top-N evaluation.
        
        Args:
            results (dict): A dictionary of results of the form 
            `{product_smiles: [reactant_smiles]}`.
        
        Returns:
            float: The top-K accuracy.
        """
        top_k_count = 0
        # Load test data
        with open(self._test_path, 'r') as f:
            targets = f.read().replace(" ", "").split("\n")
        for i, reaction in enumerate(results.values()):
            for k, reactants in enumerate(reaction):
                reactant_identical = self.check_reactant_str(reactants, targets[i])
                if reactant_identical is True:
                    if k+1 <= self._k:
                        top_k_count += 1
                    break
                elif k > self._k:
                    break
        return top_k_count / len(results)

    def check_reactant_str(self, reactant_str_1, reactant_str_2):
        """Checks two sets of reactant strings.
        
        Args:
            reactant_str_1 (list): A list of reactant SMILES strings.
            reactant_str_2 (list): A second list of reactant SMILES strings.

        Notes: 
            Reactant strings should be of the form "reactant1.reactant2.reactant3".
            Multiple components of a reactant should be seperated by a period.
            This function will work for any number and order of reactants.
        """
        r_1_list = reactant_str_1.split(".")
        r_2_list = reactant_str_2.split(".")
        if len(r_1_list) != len(r_2_list):
            return False
        else:
            for r_1 in r_1_list:
                if any(map(lambda x: self.compare_smiles(r_1, x), r_2_list)):
                    if r_1 == r_1_list[-1]:
                        return True
                    
            return False

  
class Duplicates(Evaluator):
    """Quantifies the number of duplicated reactants. This trivial metric will simply check the number of unique strings.
    """

    def __init__(self):
        """Initializes the evaluator.
        """
        super().__init__()

    def __call__(self, results: dict) -> float:
        """Calculates the diversity of the generated reactants.
        
        Args:
            results (dict): A dictionary of results of the form 
            `{product_smiles: [reactant_smiles]}`.
        
        Returns:
            float: The diversity of the generated reactants.
        """
        reactions = []
        for reaction in results.values():
            reactant_set = set()
            for reactants in reaction:
                reactant_set.add(reactants)
            reactions.append(reactant_set)
        num_unique_reactants = sum([len(x) for x in reactions])
        print(num_unique_reactants)

        total_reactions = len(list(results.values())[0]) * len(results)
        return 1-(num_unique_reactants / total_reactions)
    

def main():
    metrics = {
        "Top-10: ": TopK(10), # Maximise 
        "Duplicates:": Duplicates(), # Minimise
        "Invalid SMILES: ": InvalidSMILES(), # Minimise
        # SCScore(), # Minimise
    }
    weights = [
        3,
        1, 
        1
    ]
    with open("task1.json", 'r') as f:
        results = json.load(f)
    tot = 0
    scoring_str = ""
    for i, callable in enumerate(metrics):
        unscaled_res = round(metrics[callable](results), 2) 
        tot += weights[i] * unscaled_res
        scoring_str += callable + str(f"     {unscaled_res}") + "\n\n"
    # Perform min-max scaling between 0 and 5
    scaled_score = scale_value(tot, 0, sum(weights)) # Sum weights assumes each value is 1
    scoring_str += "Total Score: " + str(int(scaled_score)) + "\n"
    # Write to a file
    with open("scores_task_1.txt", 'w') as f:
        f.write(scoring_str)

def scale_value(value, min_val, max_val):
    scaled_value = 90 * (value - min_val) / (max_val - min_val)
    return scaled_value

if __name__ == "__main__":
    main()

def tokenize_smiles(smiles):
    import re
    pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    assert smiles == ''.join(tokens)
    return ' '.join(tokens)