import numpy as np
import scipy.sparse as sparse


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix) -> np.ndarray:
    """This function calculates the pairwise Tanimoto-similarity between rows in matrix_a and rows in matrix_b.

    For two binary fingerprints the Tanimoto-similarity is defined as:
            sim_T(fp_1, fp_2) = intersection(fp_1, fp_2) / union(fp_1, fp_2)
            sim_T(fp_1, fp_2) = intersection(fp_1, fp_2) / (sum(fp_1) + sum(fp_2) - intersection(fp_1, fp_2))
            sim_T(fp_1, fp_2) = fp_1 dot fp_2 / (fp_1 dot fp_1 + fp_2 dot fp_2 - fp_1 dot fp_2)

    This function only works for binary fingerprints and does not consider counted fingerprints.
    The tanimoto-similarity for two empty fingerprints is not covered as well.

    :returns: a matrix with the shape (n_rows(matrix_a), n_rows(matrix_b)), where all elements are between 0 and 1.

    Explanation:
        Assuming you want to compare fingerprint-matrix-A (A) with fingerprint-matrix-B (B) (fingerprints are given
        row-wise).

        Intersection:
            The intersection for two fingerprints is:
                intersection(fp_1, fp_2) = fp_1 dot fp_2
            For pairwise comparing, a matrix is constructed where element(i,j) denotes the intersection of fp_i with
            fp_j.
            Here the fingerprints are given in rows in matrices. Therefore the intersection is expressed as:
                intersection(i,j) = A(row_i) dot B(row_j) for all rows i in A and all rows j in B.

            The dot product of two matrices (O dot P) yields a matrix where the element(i,j) is the dot product of the
            row i in matrix O and column j in matrix P. Transposing a matrix (.T) turns rows into columns and vice
            versa.
            Combining both allows to express the intersection-matrix as:
                Term_intersection = A dot B.T

        Union:
            The union of two sets can be expressed as the number of elements in set_1 plus the number of elements in
            set_2 minus the intersection of set_1 and set_2. Thus the union for two fingerprints is:
                union(fp_1, fp_2) = sum(fp_1) + sum(fp_2) - intersection(fp_1, fp_2)
            The matrix for pairwise union is:
                union(i, j) = sum(A(row_i))  +  sum(B(row_j))  -  A(row_i) dot B(row_j)
            As you can see the term "sum(A(row_i))" is identical for every element in row i, whereas "sum(B(row_j))" is
            identical for each element in column j. Therefore both terms are calculated once and projected to the rest
            of the row, respective column using a vector of ones:
                union = Term_a + Term_b - Term_intersection
                Term_a = A.sum_rows dot ones(length = n_rows(B))
                Term_b = B.sum_rows dot ones(length = n_rows(A)).T
                Term_intersection = A dot B.T
            A.sum_rows and B.sum_rows are column vectors with a length equal to the number of rows. These are
            multiplied with a row-vector of ones with a length equal of the number of rows of the other matrix.
            Shapes:
             Term_a:              (n_rows(A), 1)  dot  (1, n_rows(B))
                                      -> (n_rows(A), n_rows(B))

             Term_b:            [ (n_rows(B), 1)  dot  (1, n_rows(A)) ].T
                                      -> (n_rows(B), n_rows(A)).T
                                      -> (n_rows(A), n_rows(B))

             Term_intersection:   (n_rows(A), n_cols(A))  dot  (n_cols(B), n_rows(B))
                                      -> (n_rows(A), n_rows(B))

             IF AND ONLY IF n_cols(A) == n_cols(B) (both fingerprints have the same number of bits)

        Tanimoto-similarity:
            Calculating:
                tanimoto_matrix = Term_intersection / (Term_a + Term_b - Term_intersection)
    """
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    feature_sum_a = np.array(matrix_a.sum(axis=1))
    feature_sum_b = np.array(matrix_b.sum(axis=1))
    union = feature_sum_a + feature_sum_b.T - intersection
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
