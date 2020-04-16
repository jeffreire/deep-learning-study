import numpy as np
import pandas as pd

# class rede neural para aplicacoes simples
class NeuralNetworks:

    def __init__(self, epocas, entrada, saida, peso, peso1, momento, taxa_aprendizagem):
       
        self.epocas = epocas
        self.X = entrada
        self.y = saida
        self.w =  peso
        self.w_1 = peso1
        self.momentum = momento
        self.learning_rate = taxa_aprendizagem

# Metodo que calcula a sigmoid passando por parametro a soma (perceptron)
    def sigmoid(self, soma):
        return 1 / (1 + np.exp(-soma))

# Metodo que calcula e retorna a derivada da sigmoid/
    def sigmoid_Derivada(self, sig):
        return sig * (1 - sig)

# metodo que executa a todas as funcoes da classe
    def aplications(self):
        for i in range(self.epocas):
            i = i
            
            layer_input = self.X
            # entrada ate  camada oculta
            # calcula o perceptron, passando por parametro as entradas e os pesos
            sum_perceptron = np.dot(layer_input, self.w)
            # calcula a sigmoid da camada oculta atraves da soma do perceptron
            layer_hidden = self.sigmoid(sum_perceptron)
            # camada oculta ate a camada de saida
            # calculando o perceptron passando por parametro a camada oculta e os pesos da camada oculta ate a saida
            sum_perceptron2 = np.dot(layer_hidden, self.w_1)
            # calculando a sigmoid da camada de saida passando por parametro o perceptron da camada oculta 
            layer_output = self.sigmoid(sum_perceptron2)

            # calculando os erros do resultado da rede neural
            erro_layer_output = self.y - layer_output
            # calculado a media dos erros
            mean_absolut_error = np.mean(np.abs(erro_layer_output))
            print('Error: ' + str(mean_absolut_error))

            # calculando a derivada da camada de saida
            derivada_output = self.sigmoid_Derivada(layer_output)
            # calculando o delta da camada de saida
            delta_output = erro_layer_output * derivada_output
            # realizando a transposta dos pesos da camada oculta ate a camada de saida
            peso1_traspost =  self.w_1.T
            # calculado o delta da camada de saida x a camada de entrada
            delta_output_w = delta_output.dot(peso1_traspost)
            # calculando o delta da camada oculta, somando a multiplicacao da camada de saida vezes a camada de entrada + a derivada da sigmoid
            delta_layer_hidden = delta_output_w + self.sigmoid_Derivada(layer_hidden)
            
            # calculando a camada oculta trasposta 
            layer_hidden_traspost = layer_hidden.T
            # obtendo os novos pesos da camada oculta a camada de saida
            w_news1 = layer_hidden_traspost.dot(delta_output)
            # usando a formula do backpropagation para calcular e definir um novo peso da camada oculta ate a camada de saida
            self.w_1 = (self.w_1 * self.momentum) + (w_news1 * self.learning_rate)
            
            # calculado a camada de entrada trasposta
            layer_input_transpost = layer_input.T
            # obtendo um novo peso da camada de entrada ate a camada oculta
            w_new0 = layer_input_transpost.dot(delta_layer_hidden)
            # usando a formula do backpropagation para definir um novo peso da camada de entrada ate a camada de saida
            self.w = (self.w * self.momentum) + (w_new0 * self.learning_rate)

data = pd.read_csv('Neural_Networks/Aplication/Data/base.csv')
X = data.iloc[:, [1,2,3]].values
saidas = data.iloc[:, 0].values
y = np.empty([40, 1], dtype= int)
for i in range(40):
    y[i] = saidas[i]

# random_sample(x,y) x é o total de camadas de entrada e y é a quantidade de camadas oculta
w_0 = 2*np.random.random_sample((3,3)) - 1
# (x,y) x camada escondida e y a camada de saida
w_1 = 2*np.random.random_sample((3,1)) - 1

momento = 1
taxa_aprendizagem = 0.5
epocas = 10000
rede = NeuralNetworks(epocas, X, y, w_0, w_1, momento, taxa_aprendizagem)
rede.aplications()