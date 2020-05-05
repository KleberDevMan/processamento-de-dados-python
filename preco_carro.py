# Atividade de Regressao Linear Multipla

# prever o preço de um carro com base no:
# 1. tipo de combustível
# 2. massa total do veículo
# 3. tamanho do motor
# 4. cavalos potencia

# importando bibliotecas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('carros_data.csv')

# dando uma olhada nos dados
# data

# verificando se tem dados vazios
# data.isnull().sum()

# grafico de calor para verficar variáveis correlacionadas
# f, ax = plt.subplots(figsize=(24, 21))
# sns.heatmap(data.corr(), annot=True, linewidths=10.0, ax=ax)

# variaveis independentes
# *fueltype             tipo de combustível
# *curbweight           massa total
# *enginesize           tamanho do motor
# *horsepower           cavalo-vapor (21)
X = data.iloc[:, [3, 13, 16, 21]]
# variavel dependente
y = data.iloc[:, -1].values

# codifica dados categoricos: fueltype
ct = ColumnTransformer(transformers=[(
    'encolder_fueltype', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# descosiderando uma das colunas das variáveis fictícias geradas pelo OneHotEncoder
# evitar o problema: Dump Variable trap
X = X[:, 1:]

# separar dados de testes e dados de treinamento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# regressao linear multipla
regressor = LinearRegression()

# Preparando o regressor
regressor.fit(X_train, y_train)
regressor.intercept_ # preco do carro se for zero em todas variáveis independentes
regressor.coef_ # todos coeficientes (angulo da reta) gerados para cada variavel

# Com horsepower:
# Constante: b0 = -10252.20 --> significa que se o carro tiver zero em todas variaveis independentes
# Coeficiente: b1 = -2241.94
# Coeficiente: b2 = 3.50
# Coeficiente: b3 = 71.51
# Coeficiente: b4 = 71.13
# preco = b0 + b1 * fueltype + b2 * curbweight + b3 * enginesize + b4 * horsepower
# preco = -10252.20 + -2241.94 * fueltype + 3.50 * curbweight + 71.51 * enginesize + 71.13 * horsepower

# Sem horsepower:
# Constante: b0 = -14372.43 --> significa que se o carro tiver zero em todas variaveis independentes
# Coeficiente: b1 = 331.59
# Coeficiente: b2 = 5.38
# Coeficiente: b3 = 106.23
# preco = b0 + b1 * fueltype + b2 * curbweight + b3 * enginesize
# preco = -14372.43 + 331.59 * fueltype + 5.38 * curbweight + 106.23 * enginesize

# Fazendo a predição do valor dos carros dos dados de teste
y_pred = regressor.predict(X_test)

# comparando valores previstos com os valores reais
compara_ys = np.concatenate(
    ( np.around(y_pred,1).reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Predição de um novo carro
# parametros: gasolina, massa tt do carro, peso do motor, cavalos de potência
gol = np.reshape([1, 2500, 130, 82], (-1, 4)) # tranformo em uma matriz 1x3 (linhaxcoluna)
print('preço do gol: R$', format(regressor.predict(gol)[0], '.2f'))