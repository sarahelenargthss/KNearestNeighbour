# Sara Helena Régis Theiss

from numpy.lib.nanfunctions import nanargmin
import scipy.io as scipy
from math import sqrt
from collections import Counter as ct
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np

# busca valor da característica 'indice' de dados que correspondem ao rotulo passado 
def get_dados_rotulo(dados, rotulos, rotulo, indice):
    ret = []
    for idx in range(0, len(dados)):
        if(rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])
    return ret

# cria gráfico onde cada elemento de dados é representado de acordo com seu rótulo nas coordenadas das duas dimensões escolhidas
def visualiza_pontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()
    ax.scatter(get_dados_rotulo(dados, rotulos, 1, d1), get_dados_rotulo(dados, rotulos, 1, d2), c='red' , marker='^')
    ax.scatter(get_dados_rotulo(dados, rotulos, 2, d1), get_dados_rotulo(dados, rotulos, 2, d2), c='blue' , marker='+')
    ax.scatter(get_dados_rotulo(dados, rotulos, 3, d1), get_dados_rotulo(dados, rotulos, 3, d2), c='green', marker='.')
    plt.show()

# calcula a acurácia entre os rótulos previstos e os rótulos de teste
def acuracia(previstos, teste):
    # calcula o número de rótulos previstos corretamente
    num_corretos = 0
    for i in range(len(previstos)):
        num_corretos += 1 if previstos[i] == teste[i,0] else 0

    total_num = len(teste)
    acuracia = num_corretos / total_num * 100
    return acuracia

# calcula a distância entre os elementos a e b, sendo n a dimensão deles
def dist(a, b, n):
    soma_dimensoes = 0

    try:
        for i in range(n):
            soma_dimensoes += (b[i] - a[i]) ** 2
        return sqrt(soma_dimensoes)
    except IndexError:
        print('Dimensão inválida para os elementos!')

# faz a normalização dos dados passados
def normalizacao(dados):
    n = len(dados[0]) # número de colunas/características
    valores_fixos = []

    # salva os valores que serão fixos para cada coluna, como o valor mínimo encontrado
    # e o cálculo do valor mínimo subtraído do valor máximo 
    for i in range(n):
        x = [row[i] for row in dados]
        valores = []
        valores.append(min(x)) # 𝑀𝑖𝑛(𝑋)
        valores.append(max(x) - valores[0]) # 𝑀𝑎𝑥 𝑋 −𝑀𝑖𝑛(𝑋)
        valores_fixos.append(valores) # valores fixos da coluna i de dados
    
    novos_dados = [] # cria nova matriz de dados para não modificar a matriz original
    for i in range(len(dados)):
        elemento = []
        for j in range(n):
            # 𝑁(𝑋) = [𝑋−𝑀𝑖𝑛(𝑋)] / 𝑀𝑎𝑥 𝑋 −𝑀𝑖𝑛(𝑋)
            elemento.append((dados[i][j] - valores_fixos[j][0]) / valores_fixos[j][1])
        novos_dados.append(elemento)
    
    return np.array(novos_dados)


# classifica cada valor de teste de acordo com os exemplos de treinamento mais próximos a eles
# adicionado parâmetro normalizar, que quando True habilita a normalização dos dados 
def meu_knn(dados_train, rotulo_train, dados_teste, k, normalizar):
    rotulos = []

    if normalizar:
        # aplica normalização aos dados
        dados_teste = normalizacao(dados_teste)
        dados_train = normalizacao(dados_train)

    for i in range(len(dados_teste)):
        distancia_teste_train = []
        for j in range(len(dados_train)):
            # calcula a distância entre o elemento i de teste e o elemento j de treinamento
            distancia_teste_train.append(dist(dados_teste[i], dados_train[j], len(dados_teste[i])))

        # ordena a lista de distância e rótulos juntos para que fiquem com os mesmos índices nos 
        # elementos correspondentes; então, seleciona a lista com os rótulos já ordenados
        rotulos_ordenados = [y for x, y in sorted(zip(distancia_teste_train, rotulo_train))]

        # salva somente a coluna com os rótulos, que é o valor interessante para o problema
        rotulos_ordenados = [row[0] for row in rotulos_ordenados]

        # seleciona os k primeiros elementos dos rótulos, calcula a quantidade de ocorrências 
        # para cada valor e pega o valor mais comum encontrado
        rotulos.append(ct(rotulos_ordenados[:k]).most_common(1)[0][0])
    
    return rotulos

# carrega grupo de dados
mat = scipy.loadmat('grupoDados3.mat')
grupo_test = mat['grupoTest']
grupo_train = mat['grupoTrain']
test_rots = mat['testRots']
train_rots = mat['trainRots']

# Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
# Aplicando o kNN com k igual a 1, obteve-se uma acurácia de 70%
rotulo_previsto_1 = meu_knn(grupo_train, train_rots, grupo_test, 1, False)
valor_acuracia_1 = acuracia(rotulo_previsto_1, test_rots)
print('Q3.1 - Acurácia obtida: ', valor_acuracia_1, end='%\n\n')
visualiza_pontos(grupo_test, rotulo_previsto_1, 0, 1)

# Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor.
#       Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e
#       por quê.
# O resultado ficou bem menor, pois apenas um vizinho mais próximo não foi suficiente para fazer a classificação dos
# elementos de teste. Abaixo são calculados novos valores para k, a fim de obter a maior acurácia possível.
# Com k igual a 10, pôde-se alcançar a acurácia de 92%. Isso foi feito para que cada elemento de teste seja classificado
# considerando um contexto maior da vizinhança e não apenas o mais próximo.

maior_acuracia = 0
melhor_k = 0
melhores_rotulos = []
for i in range(1, len(grupo_train)):
    rotulo_previsto = meu_knn(grupo_train, train_rots, grupo_test, i, False)
    valor_acuracia = acuracia(rotulo_previsto, test_rots)

    if valor_acuracia > maior_acuracia:
        maior_acuracia = valor_acuracia
        melhor_k = i
        melhores_rotulos = rotulo_previsto
print('Q3.2 - Acurácia obtida: ', maior_acuracia, end='%\n')
print('K:', melhor_k, end='\n')
visualiza_pontos(grupo_test, rotulo_previsto_1, 0, 1)