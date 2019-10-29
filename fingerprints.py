import abc
from typing import *

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import scipy.sparse as sparse
from bidict import bidict


class Fingerprint(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def n_bits(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError


class FoldedMorganFingerprint(Fingerprint):
    def __init__(self, n_bits=2048, diameter: int = 2, use_features=False):
        super().__init__()
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {n_bits})")
        if isinstance(diameter, int) and diameter >= 0:
            self._diameter = diameter
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {diameter})")

        self._use_features = use_features

    @property
    def n_bits(self):
        return self._n_bits

    @property
    def diameter(self):
        return self._diameter

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        pass

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        fingerprints = []
        for mol in mol_obj_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.diameter, useFeatures=self._use_features)
            fingerprints.append(sparse.csr_matrix(fp))
        return sparse.vstack(fingerprints)

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)


class UnfoldedMorganFingerprint(Fingerprint):
    """Transforms smiles-strings into unfolded bit-vectors based on Morgan-fingerprints [1].
    Features are mapped to bits based on the amount of molecules they occur in.

    Long version:
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.

    References:
            [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    def __init__(self, counted: bool = False, diameter: int = 2, use_features: bool = False):
        """ Initializes the class

        :param counted: if False, bits are binary: on if present in molecule, of if not present
                        if True, bits are positive integers and give the occurrence of their respective features in the
                        molecule
        :param diameter: diameter of the circular fingerprint [1]
        :param use_features:

        References:
            [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
        """
        super().__init__()
        self._bit_mapping = None
        if isinstance(diameter, int) and diameter >= 0:
            self._diameter = diameter
        else:
            raise ValueError(f"Not a positive integer: {diameter}")

        self._counted = counted
        self.use_features = use_features

    @property
    def counted(self):
        return self._counted

    @property
    def diameter(self):
        return self._diameter

    @property
    def n_bits(self):
        if self._bit_mapping:
            return len(self._bit_mapping)
        else:
            raise ValueError("Length not determined yet!")

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        self._create_mapping(mol_iterator)

    def _gen_features(self, mol_obj: Chem.Mol) -> Dict[int, int]:
        return AllChem.GetMorganFingerprint(mol_obj, self.diameter, useFeatures=self.use_features).GetNonzeroElements()

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_fp_list = [self._gen_features(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(mol_fp_list)
        return self._transform(mol_fp_list)

    def transform(self,  mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_iterator)

    def _transform(self, mol_fp_list: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        if self._counted:
            for i, mol_fp in enumerate(mol_fp_list):
                for feature, count in mol_fp.items():
                    data.append(count)
                    rows.append(self._bit_mapping[feature])
                    cols.append(i)
        else:
            for i, mol_fp in enumerate(mol_fp_list):
                data.extend([1] * len(mol_fp))
                rows.extend([self._bit_mapping[feature] for feature in mol_fp.keys()])
                cols.extend([i] * len(mol_fp))
        return sparse.csr_matrix((data, (cols, rows)), shape=(len(set(cols)), self.n_bits))

    def _create_mapping(self, molecule_features: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]):
        unraveled_features = [f for f_list in molecule_features for f in f_list.keys()]
        unique_features = set(unraveled_features)
        feature_order = sorted(unique_features, key=lambda f: unraveled_features.count(f), reverse=True)
        self._bit_mapping = bidict(zip(feature_order, range(len(feature_order))))


if __name__ == "__main__":
    from supporting_functions import construct_check_mol_list
    # noinspection SpellCheckingInspection
    test_smiles_list = ["c1ccccc1",
                        "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
                        "c1cc(ccc1C2CCNCC2COc3ccc4c(c3)OCO4)F",
                        "c1c(c2c(ncnc2n1C3C(C(C(O3)CO)O)O)N)C(=O)N",
                        "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl",
                        "CN(C)c1c2c(ncn1)n(cn2)C3C(C(C(O3)CO)NC(=O)C(Cc4ccc(cc4)OC)N)O",
                        "CC12CCC(CC1CCC3C2CC(C4(C3(CCC4C5=CC(=O)OC5)O)C)O)O",
                        ]
    test_mol_obj_list = construct_check_mol_list(test_smiles_list)
    ecfp2_1 = UnfoldedMorganFingerprint()
    fp1 = ecfp2_1.fit_transform(test_mol_obj_list)
    print(fp1.shape)
    ecfp2_2 = FoldedMorganFingerprint()
    fp2 = ecfp2_2.fit_transform(test_mol_obj_list)
    print(fp2.shape)