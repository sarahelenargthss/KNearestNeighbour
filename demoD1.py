import scipy.io as scipy
from math import sqrt
from collections import Counter as ct

# calcula a distância entre os elementos a e b, sendo n a dimensão deles
def distancia(a, b, n):
    soma_dimensoes = 0

    try:
        for i in range(n):
            soma_dimensoes += (b[i] - a[i]) ** 2
        return sqrt(soma_dimensoes)
    except IndexError:
        print('Dimensão inválida para os elementos!')

# classifica cada valor de teste de acordo com os exemplos de treinamento mais próximos a eles
def meuKnn(dados_train, rotulo_train, dados_teste, k):
    rotulos = []
    for i in range(len(dados_teste)):
        distancia_teste_train = []
        for j in range(len(dados_train)):
            # calcula a distância entre o elemento i de teste e o elemento j de treinamento
            distancia_teste_train.append(distancia(dados_teste[i], dados_train[j], dados_teste[i].size))

        # ordena a lista de distância e rótulos juntos para que fiquem com os mesmos índices nos 
        # elementos correspondentes; então, seleciona a lista com os rótulos já ordenados
        rotulos_ordenados = [y for x, y in sorted(zip(distancia_teste_train, rotulo_train))]

        # salva somente a coluna com os rótulos, que é o valor interessante para o problema
        rotulos_ordenados = [row[0] for row in rotulos_ordenados]

        # seleciona os k primeiros elementos dos rótulos, calcula a quantidade de ocorrências 
        # para cada valor e pega o valor mais comum encontrado
        rotulos.append(ct(rotulos_ordenados[:k]).most_common(1)[0])
    
    return rotulos


mat = scipy.loadmat('grupoDados1.mat')

grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']

rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, 1)
estaCorreto = rotuloPrevisto == testRots
numCorreto = ct([row[0] for row in estaCorreto])[True]
totalNum = len(testRots)
acuracia = numCorreto / totalNum

print(acuracia)