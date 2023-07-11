# Performing preprocessing on the data
import re
import pandas as pd
from rdkit import Chem

class Preproces:

    def get_smiles_from_df(self, df:pd.DataFrame) -> tuple:
        reactions = df.iloc[:,-1].values
        products, reactants = [], []
        for reaction in reactions:
            reac, pro = reaction.split('>>')
            try:
                pro_mol, reac_mol = Chem.MolFromSmiles(pro), Chem.MolFromSmiles(reac)
                [a.SetAtomMapNum(0) for a in pro_mol.GetAtoms()]
                [a.SetAtomMapNum(0) for a in reac_mol.GetAtoms()]
                product_smi = Chem.MolToSmiles(pro_mol, canonical=True, isomericSmiles=True)
                reactant_smi = Chem.MolToSmiles(reac_mol, canonical=True, isomericSmiles=True)
                products.append(self.tokenize_smiles(product_smi)), reactants.append(self.tokenize_smiles(reactant_smi))
            except:
                continue
            return products, reactants
        

    @staticmethod
    def tokenize_smiles(smiles):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        assert smiles == ''.join(tokens)
        return ' '.join(tokens)
    
