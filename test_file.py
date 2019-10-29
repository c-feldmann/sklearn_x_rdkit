import unittest

from fingerprints import UnfoldedMorganFingerprint

# noinspection SpellCheckingInspection
smiles_list = ["c1ccccc1",
               "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
               "c1cc(ccc1C2CCNCC2COc3ccc4c(c3)OCO4)F",
               "c1c(c2c(ncnc2n1C3C(C(C(O3)CO)O)O)N)C(=O)N",
               "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl",
               "CN(C)c1c2c(ncn1)n(cn2)C3C(C(C(O3)CO)NC(=O)C(Cc4ccc(cc4)OC)N)O",
               "CC12CCC(CC1CCC3C2CC(C4(C3(CCC4C5=CC(=O)OC5)O)C)O)O",
               ]


class ConstructingFingerprints(unittest.TestCase):
    def test_independence_of_constructing(self):
        ecfp2_1 = UnfoldedMorganFingerprint()
        fp1 = ecfp2_1.fit_transform(smiles_list)
        ecfp2_2 = UnfoldedMorganFingerprint()
        ecfp2_2.fit(smiles_list)
        fp2 = ecfp2_2.transform(smiles_list)
        self.assertTrue((fp1 != fp2).nnz == 0)


if __name__ == '__main__':
    unittest.main()
