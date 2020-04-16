import pandas as pd
import numpy as np
import random as rd

class Color_learning:
    def __init__(self, feature, labls, entradas, oculta, saida, inst, lr, ephoca, ohl):
        self.feature = feature
        self.labels = labls
        self.input =  entradas
        self.output = saida
        self.hidden = oculta
        self.intances = inst
        self.learning_rater = lr
        self.ephoca = ephoca
        self.one_hot_labels = ohl

    @staticmethod
    def sigmoid(sm):
        return 1/(1 + np.exp(-sm))

    def sigmoid_der(self, sm):
        return self.sigmoid(sm)*(1 - self.sigmoid(sm))

    @staticmethod
    def reluFunction(soma):
        return soma * (soma > 0)

    @staticmethod
    def softmax(sm):
        ex = np.exp(sm)
        return ex / ex.sum(axis=1, keepdims=True)

    @staticmethod
    def saveFile(wh, wo, bh, bo):
        np.savetxt('W_hidden', wh)
        np.savetxt('W_output', wo)
        np.savetxt('Bias_hidden', bh)
        np.savetxt('Bias_output', bo)
        pass
    
    def pesos_and_bias(self):
        self.w_hidden = np.array([
            [3.745401188473624909e-01,2.523058828614708204e+00,7.319939418114050911e-01,2.389864374595511709e+00,-2.569265545780381022e+00],
            [1.559945203362026467e-01,-6.273783669562087439e-01,8.661761457749351800e-01,5.685504174331969196e+00,2.199633592824242090e+00],
            [2.058449429580244683e-02,3.580545654210289808e+00,8.324426408004217404e-01,-6.914150925823431271e-01,5.734765646094067471e+00]])
        
        self.bias_hidden = np.array([[-5.713801657541415224e-01,1.599035767953929543e+00,-2.612549012693601291e+00,4.021618985447205752e+00,4.077521552978897290e+00]])
        
        self.w_output = np.array([[1.394938606520418345e-01,2.921446485352181544e-01,3.663618432936917024e-01,4.560699842170359286e-01,7.851759613930135995e-01],                            
            [-1.357900622441514438e+00,1.062887835078976995e+00,1.666834295713892411e+00,2.228486176073957506e-02,5.662116839433314341e-01],
            [1.705241236872915289e-01,6.505159298527951606e-02,9.488855372533332444e-01,9.656320330745593594e-01,8.083973481164611341e-01],
            [1.410244732192115658e+00,1.849088662447048170e+00,-1.080324002991959631e+00,5.723329450902929494e-02,-5.875330478799644096e-01],    
            [1.349525241404493592e+00,-1.235038872119687881e+00,2.177943211965753800e+00,-3.732755317015576391e-02,1.050860711788916130e-01]])
                             
        self.bias_output = np.array([[-1.524395046956204203e+00,8.073278612459058312e-01,-2.031484697797881545e+00,-1.504860077798268136e+00,-1.525953782137729475e-01]])

    def initial(self):
        for i in range(self.ephoca):
            i = i
            sum_sinapse1 = np.dot(self.feature, self.w_hidden) + self.bias_hidden
            layer_hidden = self.reluFunction(sum_sinapse1)

            sum_sinapse2 = np.dot(layer_hidden, self.w_output) + self.bias_output
            layer_output = self.softmax(sum_sinapse2)

            # Backward propagation
            error_layer_output = layer_output - self.one_hot_labels
            layer_hidden_traspost = layer_hidden.T
            delta_input_w_h = np.dot(layer_hidden_traspost, error_layer_output)

            w_output_traspost = self.w_output.T
            delta_output_xW_h = np.dot(error_layer_output, w_output_traspost)
            derivada_layer_hidden = self.reluFunction(sum_sinapse1)
            
            feature_transpost = self.feature.T
            delta_hiddenXw_o = np.dot(feature_transpost, derivada_layer_hidden * delta_output_xW_h)
            delta_layer_hidden = delta_output_xW_h * derivada_layer_hidden

            self.w_hidden -= self.learning_rater * delta_hiddenXw_o
            self.bias_hidden -= self.learning_rater * delta_layer_hidden.sum(axis=0)

            self.w_output -= self.learning_rater * delta_input_w_h
            self.bias_output -= self.learning_rater * error_layer_output.sum(axis=0)
            
            loss = np.sum(-self.one_hot_labels * np.log(layer_output))
            print('Loss function value:' + str(loss))

red = []
green = []
blue = []
for i in range(10):
    red.append([(rd.randint(200, 255)/255), (rd.randint(100, 180)/255), (rd.randint(0, 99)/255)])
    green.append([(rd.randint(0, 99)/255), (rd.randint(200, 255)/255), (rd.randint(100, 180)/255)])
    blue.append([(rd.randint(0, 99)/255), (rd.randint(100, 180)/255), (rd.randint(200, 255)/255)])

feature_set = np.vstack([green, red, blue])
labels = np.array([0]*10 + [1]*10 + [2]*10)

one_hot_labels = np.zeros((30, 5))
for i in range(30):
    one_hot_labels[i, labels[i]] = 1

instances = feature_set.shape[0]
entradas = feature_set.shape[1]
camadas_ocultas = 5
camadas_saidas = 5
ephocas = 1000
learning_rate = 1e-4

color = Color_learning(feature_set, labels, entradas, camadas_ocultas, camadas_saidas, instances, learning_rate, ephocas, one_hot_labels)
color.pesos_and_bias()
color.initial()

