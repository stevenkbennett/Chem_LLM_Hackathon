# This script is used to evaluate the model on the test set, and is used to generate results. 
# The Evaluator class can be used to evalate the model on the test set, and generate results.
from rdkit import Chem
import numpy as np

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
            Chem.MolFromSmiles(smi)
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
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, kekuleSmiles=True) for smi in [smi1, smi2]
        ]
        return smi1 == smi2

class RoundTrip(Evaluator):
    """Performs a round-trip evaluation."""


class TopK(Evaluator):
    """Performs a top-N evaluation."""
    def __init__(self, k):
        """Initializes the evaluator.
        
        Args:
            k (int): The number of predictions to consider.
        """
        self._k = k
        super().__init__()

    def __call__(self, results: dict, targets: dict) -> float:
        """Performs a top-N evaluation.
        
        Args:
            results (dict): A dictionary of results of the form 
            `{product_smiles: [reactant_smiles]}`.

            targets (dict): A dictionary of targets of the form 
            `{product_smiles: [reactant_smiles]}`.
        
        Returns:
            float: The top-K accuracy.
        """
        top_k_count = 0
        for target, reaction in results.items():
            for k, reactants in enumerate(reaction):
                reactant_identical = self.check_reactant_str(reactants, targets[target])
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
        
