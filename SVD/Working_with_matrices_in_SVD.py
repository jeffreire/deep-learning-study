import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

# Calss that run the conversion
class Matrix:
    def __init__(self, mtr):
        self.matrix = mtr
    
    # method that apply technique SVD in matrices, returning tree matrices
    def _applying_SVD(self):
        self.U, self.D, self.V = np.linalg.svd(self.matrix)
        return self.U, self.D, self.V
    
    # method that reconsting the matrices returned by technique SVD
    def _reconsting_original_matrix(self):
        Sigma = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))
        Sigma[:self.matrix.shape[1], :self.matrix.shape[1]] = np.diag(self.D)
        B = self.U.dot(Sigma.dot(self.V))
        print(B)

    #method that reconstif the matrices in different dimensions, passing by parameter a list of dimensions
    # returning a list with the results
    def _reconsting_matrices_in_diferent_dimensoes(self, dimensions):
        x = []
        for i in dimensions:
            reconsting = np.matrix(self.U[:, :i]) * np.diag(self.D[:i]) * np.matrix(self.V[:i,:])
            x.append(reconsting)
        return x

# m = np.array([[2,3,4,5,6], [2,3,4,5,6], [2,3,4,5,6]])
# mt = Matrix(m)
# w = mt._applying_SVD()
# print(w[0])
# print(w[1])
# print(w[2])
# l = [1,2,3]
# print('a: {}'.format(mt._reconsting_matrices_in_diferent_dimensoes(l)))