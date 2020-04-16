# '-------------------------- Adjustment of errors, by the logical operator: And -----------------------'
import numpy as np

# data of input 
x = np.array([ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ]])

# data of exit
y = np.array([ 0, 0, 0, 1 ])

# pesos
w = np.array([ 0.0, 0.0 ])

# learning rate
learning_rate = 0.1

# method that compute neuron returns on or of
def _stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

# method that campute a exit 
def _calculate_exite(records):
    s = records.dot(w)
    return _stepFunction(s)

# method that train the data to reach 100% correction, changing the weights
def _train():
    total_error = 1

    while total_error != 0:

        total_error = 0

        for i in range(len(y)):
            y_calculada = _calculate_exite(np.asarray(x[i]))
            erro = abs(y[i] - y_calculada)
            total_error += erro
            
            for j in range(len(w)):
                w[j] = w[j] + (learning_rate * x[i][j] * erro)
                print('Updated weight: ' + str(w[j]))

            print('Total errors: ' + str(total_error))

_train()
print('Trained Neural Network!')