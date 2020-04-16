import numpy as np

# Calss that performs algorithm Perceptron passing through input and weight parameter
class Perceptron_manual:
    def __init__(self, x, w):
        self.X = x
        self.W = w
    
    # method that sum the inputs and weights pasing througt the number of entries
    def SUM(self):
        self.s = 0
        for i in range(len(self.X)):
            self.s += self.X[i] * self.W[i]
        return self.s

    # method that neuron returns on or of
    def Function(self, s):
        if s <= 1:
            return 1
        return 0

# Calss that performs algorithm Perceptron passing through input and weight parameter
class Perceptron_with_lib:
    def __init__(self, x, w):
        self.X = x
        self.W = w

    # method that sum the inputs and weights dot product scalar
    def Sum(self):
        return self.X.dot(self.W)

 # method that neuron returns on or of
    def Function(self, s):
        if s <= 1:
            return 1
        return 0

# e = np.array([1,7,5])
# p = np.array([0.8, 0.1, 0])

# per = Perceptron_with_lib(e, p)
# s = per.Sum()
# print('soma: {}, funcao: {}'.format(s, per.Function(s)))