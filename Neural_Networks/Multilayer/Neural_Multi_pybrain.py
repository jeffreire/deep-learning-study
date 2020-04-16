# from pybrain.datasets import SupervisedDataSet
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.structure.modules import SoftmaxLayer
# from pybrain.structure.modules import SigmoidLayer
# from pybrain.tools.s import buildNetwork

# (x,y,z) = x é a quantidade de camadas de entrada, y é a quantidade de camada oculta, z é a quantidade de camada de saida
# rede = buildNetwork(3,3,1)

# (x,y) sao os previsores, dois atributos e uma class
# base = SupervisedDataSet(2, 1)

# queremos dizer que a entrada sera de (0,0) e queremos obter uma saida de (0, )
# base.addSample((0,0),(0, ))
# base.addSample((0,1),(1, ))
# base.addSample((1,0),(0, ))
# base.addSample((1,1),(0, ))
# print(base['input'])

# treinamos a nossa rede passando por parametro rede, basa, taxa de aprendizagem e momento
# treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01,
#                               momentum = 0.06)

# o for é quantas epocas iremos calcular os nossos pesos
# for i in range(1, 30000):
#     erro = treinamento.train()
#     if i % 1000 == 0:
#         print("Erro: %s" % erro)
        