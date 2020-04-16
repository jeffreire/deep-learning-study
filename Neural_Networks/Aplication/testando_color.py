import numpy as np
import pandas as pd


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def save(w_hidden, w_output, bias_hidden, bias_output):
    np.savetxt('w_hidden', w_hidden)
    np.savetxt('w_output', w_output)
    np.savetxt('bias_hidden', bias_hidden)
    np.savetxt('bias_output', bias_output)
    pass

# Gerando os dados para treinar a rede neural
green = np.random.rand(10, 3) + np.array([0,255,0])
red = np.random.rand(10, 3) + np.array([255,0,0])
blue = np.random.rand(10, 3) + np.array([0,0,255])
black = np.random.rand(10, 3) + np.array([0,0,0])
white = np.random.rand(10, 3) + np.array([255,255,255])

# juntando as arrays de treinamento em uma so 
feature_set = np.vstack([green, red, blue, black, white])

# atribuindo uma identidade para cada entrada, para obertermos a saida
labels = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
# modificando todos os arrays para zero
one_hot_labels = np.zeros((50, 5))
# gerando 1000, 0100, 0010, 0001 para cada identidade
for i in range(50):
    one_hot_labels[i, labels[i]] = 1

# atribuindo a variavel a quantidade de dados em cada entrada
instances = feature_set.shape[0]
# atribuindo a varival o numero total de entradas
entradas = feature_set.shape[1]
# informando a quantidade total de camadas ocultas
camadas_ocultas = 4
# informando a camada de saida
camadas_saidas = 5
# fixando os pesos aleatoris para cada epoca
np.random.seed(42)

y = pd.read_csv('Neural_Networks/Aplication/Peso_bias_train/w_hidden')
y = y.iloc[:,:].values
y = np.array(y)

# gerando os peso para a sinapse entre a camada de entrada a camada oculta
w_hidden = y

b1 = pd.read_csv('Neural_Networks/Aplication/Peso_bias_train/bias_hidden')
b1 = b1.values

# geranado o bias para a camada oculta
bias_hidden = b1.T

x = pd.read_csv('Neural_Networks/Aplication/Peso_bias_train/w_output')
x = x.iloc[:,:].values
x = np.array(x)
# gerando os pesos para as sinapses entre a camada oculta e a camada de saida
w_output = x

b2= pd.read_csv('Neural_Networks/Aplication/Peso_bias_train/bias_output')
b2 = b2.values

# gerando o bias na camada de saida
bias_output = b2.T

# definindo  a taxa de apredizagem
learning_rate = 1e-4

# coletetando os erros
error_cost = []

for epoch in range(10):
# calculando a sinapse  entre a cmada de entrada e a camada oculta com o bias
    sum_sinapse1 = np.dot(feature_set, w_hidden) + bias_hidden
# obtendo a camada oculta com a funcao de ativacao sigmoid
    layer_hidden = sigmoid(sum_sinapse1)

# calculando a sinapse entre a camada oculta e a camada final
    sum_sinapse2 = np.dot(layer_hidden, w_output) + bias_output
#calculando o resultado final usando a funcao de ativacao sofmax 
    layer_output = softmax(sum_sinapse2)

########## Back Propagation'''

# obtendo o erro ao gerar o resultado na camada oculta
    error_layer_output = layer_output - one_hot_labels
# gerando a matriz transposta da camada oculta
    layer_hidden_transpost = layer_hidden.T
# calculamdo o delta da camada de entrada em relacao ao erro obtido
    delta_input_w_h = np.dot(layer_hidden_transpost, error_layer_output)
# gerando a matriz transposta da camada de saida
    w_output_traspost = w_output.T
# calculando o delta da camada de sainda em relacao ao erro obtido
    delta_outputXW_h = np.dot(error_layer_output, w_output_traspost)

# calacularando a derivada da camada oculta atraves da sinapse 1
    derivada_layer_hidden = sigmoid_der(sum_sinapse1)
# gerando a transpost da matriz de entrada
    feature_set_traspost = feature_set.T
# calculando o delta da cmada oculta em relacao ao erro obtido dos pessos
    delta_hiddenXw_O = np.dot(feature_set_traspost, derivada_layer_hidden * delta_outputXW_h)
# calculando o denta da camada oculta atraves da derivada da mesma cmada
    delta_layer_hidden = delta_outputXW_h * derivada_layer_hidden

# utilizando a formula back propagation para calcular um novo peso atualizado da camada de entrad a camada oculta
    w_hidden -= learning_rate * delta_hiddenXw_O
# atualizando o bias 
    bias_hidden -= learning_rate * delta_layer_hidden.sum(axis=0)
# atualizando o peso entre a camada oculta e a camada de saida
    w_output -= learning_rate * delta_input_w_h
# atualizando bias
    bias_output -= learning_rate * error_layer_output.sum(axis=0)
# usando a entropia cruzana para calcular a margem de erro 
    loss = np.sum(-one_hot_labels * np.log(layer_output))
    print('Loss function value: ', loss)
    error_cost.append(loss)

print('Finalizado')
