import numpy as np
import scipy.sparse as sparse


def tanimoto_from_sparse(fp_matrix1: sparse.csr_matrix, fp_matrix2: sparse.csr_matrix) -> np.matrix:
    """Calculates the Tanimoto-similarity
    :returns a matrix with a pairwise comparison of vectors in fingerprint 1 with vectors of fingerprint 2
    """
    intersection = fp_matrix1.dot(fp_matrix2.transpose())
    sum_fp1 = fp_matrix1.sum(axis=1).dot(np.ones((1, fp_matrix2.shape[0])))
    sum_fp2 = fp_matrix2.sum(axis=1).dot(np.ones((1, fp_matrix1.shape[0]))).transpose()
    union = sum_fp1 + sum_fp2 - intersection
    return intersection / union


if __name__ == "__main__":
    fp1 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 0, 0]]
                                     )
                            )
    fp2 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 1, 0],
                                      [1, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      ]
                                     )
                            )
    sim = tanimoto_from_sparse(fp1, fp2)
    print(type(sim))
    print(isinstance(sim, np.matrix))
    print(sim.shape)
    print(sim)
