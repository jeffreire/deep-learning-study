import numpy as np

# metodo que retorna o resultado da sigmoid passando por parametro a soma do perceptron
def sigmoid(s):
    return (1 / (1 + np.exp(-s)))

# metodo que retorna o resultado da derivada da sigmoide passando por parametro a sigmoide
def sigmoid_Derivative(sig):
    return sig * (1 - sig)

# array de entrada
X = np.array([[0,0], 
              [0,1], 
              [1,0], 
              [1,1]])

# array de saida
y = np.array([ [0], [1], [1], [0] ])

# pesos aleatorios
w = 2*np.random.random_sample((2,3)) -1 
w1 = 2*np.random.random_sample((3,1)) -1 
print(w)

# epoca, quantas vezes o calculo sera executado do rede neural
epoch = 100

# taxa de aprendizagem pre estabelecida
learning_rate = 0.5

# variavel momento para o calculo do backpropagation
momentum = 1

# for que roda perante do valor da epoca
for i in range(epoch):

    # passo 1 : Representando os dados de entrada
    layer_input = X

    # passo 2 : Soma da sinapse dos dados de entrada multiplicado pelos pesos
    sum_sinapse = np.dot(layer_input, w)

    # passo 3 : Caculando e obtendo a camada oculta 
    layer_hidden = sigmoid(sum_sinapse)

    # passo 4  : calculando a sinapse passando por parametro a camada oculta e o peso1
    sum_sinapse_1 = np.dot(layer_hidden, w1)

    # passo 5 : calculando a camada de saida passando por parametro a soma da sinapse da camada oculta
    layer_output = sigmoid(sum_sinapse_1)

    # passo 6 : calculando o erro da camada de saida atraves da formula simples
    error_layer_output = y - layer_output

    # passo 7 : obtendo a media de todos os erros
    absolute_mean_of_errors = np.mean(np.abs(error_layer_output))
    print('Error:' + str(absolute_mean_of_errors))

    # passo 8 : calculando a derivada dos dados de saida passando por paramtro os dados de saida
    derivative_output = sigmoid_Derivative(layer_output)

    # passo 9 : calculando o delta dos erros da camada de saida
    delta_output = error_layer_output * derivative_output
    
    # passo 10 : obtendo a transpota do array do peso 1
    w1_Transpost = w1.T

    # passo 11 : calculando o delta da saida x a trasposta do peso1
    delta_output_w = delta_output.dot(w1_Transpost)

    # passo 12 : obtendo o delta da camada oculta
    delta_layer_hidden = delta_output_w * sigmoid_Derivative(layer_hidden)
    
    # passo 13 : obtendo a camada oculta trasposta 
    layer_hidden_traspost = layer_hidden.T

    # passo 14 : obtendo um novo peso atraves da camada oculta trasposta
    w_new_1 = layer_hidden_traspost.dot(delta_output)

    # passo 15 : utilizando a formula backpropagation
    w1 = (w1 * momentum) + (w_new_1 * learning_rate)
    
    # passo 16 : obtendo a transposta da camada de entrada
    layer_input_traspost = layer_input.T

    # passo 17 :  obtendo pesos novos atraves da transposta da camada de entrada x o delta da camada oculta
    w_new_0 = layer_input_traspost.dot(delta_layer_hidden)
    
    # passo 18 : calculando o backpropagation
    w = (w * momentum) + (w_new_0 * learning_rate)