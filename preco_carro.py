# Atividade de Regressao Linear Multipla

# prever o preço de um carro com base no:
# 1. tipo de combustível
# 2. massa total do veículo
# 3. tamanho do motor
# 4. cavalos potencia

# importando bibliotecas
from math import sqrt  # calulos matematicos
from sklearn.metrics import mean_squared_error # usada para calcular acurácia do modelo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # pegar estatísticas do modelo

data = pd.read_csv('carros_data.csv')

# dando uma olhada nos dados
data

# verificando se tem dados vazios
data.isnull().sum()

# grafico de calor para verficar variáveis correlacionadas
f, ax = plt.subplots(figsize=(24, 21))
sns.heatmap(data.corr(), annot=True, linewidths=10.0, ax=ax)

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
regressor.intercept_  # preco do carro se for zero em todas variáveis independentes
# todos coeficientes (angulo da reta) gerados para cada variavel
regressor.coef_

# Com horsepower:
# Constante: b0 = -10252.20 --> significa que se o carro tiver zero em todas variaveis independentes
# Coeficiente: b1 = -2241.94
# Coeficiente: b2 = 3.50
# Coeficiente: b3 = 71.51
# Coeficiente: b4 = 71.13
# preco = b0 + b1 * fueltype + b2 * curbweight + b3 * enginesize + b4 * horsepower
# preco = -10252.20 + -2241.94 * fueltype + 3.50 * curbweight + 71.51 * enginesize + 71.13 * horsepower

# Fazendo a predição do valor dos carros dos dados de teste
y_pred = regressor.predict(X_test)

# comparando valores previstos com os valores reais
compara_ys = np.concatenate(
    (np.around(y_pred, 1).reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Predição de um novo carro
# parametros: gasolina, massa tt do carro, peso do motor, cavalos de potência
# tranformo em uma matriz 1x3 (linhaxcoluna)
gol = np.reshape([1, 2500, 130, 82], (-1, 4))
print('preço do gol: U$', format(regressor.predict(gol)[0], '.2f'))


# Acurácia usando 4 variáveis
# Mean Squared Error: Erro ao quadrado médio
mse = round((mean_squared_error(y_test, y_pred))/100, 2)
# Root Mean Squared Error: Raiz quadrada do Erro ao quadrado médio
rmse = round((sqrt(mse))/100, 2)

# pergunta que fica:
# O uso do horsepower almentou a acuracia do meu modelo de predição?

# ------------------------------

# Regressão Linear Múltipla: Backward Elimination para otimização do modelo
# Passo 1: Selecionar o nível de significancia (NS = 0.05)
# Passo 2: Construir o modelo completo
# Passo 3: Ir considerando o preditor com o p-value mais alto e:
#           Se p > NS, vou para o passo 4, senão TERMINA.
# Passo 4: Retirar o preditor com p-value > NS = 0.05
# Passo 5: Construir o modelo sem esse preditor

# Crias as variáveis independentes já convertendo às categóricas

data0 = data[['fueltype', 'curbweight', 'enginesize', 'horsepower', 'carwidth', 'price']]
data1 = pd.get_dummies(data0)
X1 = data1.drop(['price'], axis=1)
y1 = data1.price

# adiciona contante b0 na minha lista de variaveis independentes
X1 = sm.add_constant(X1)
# cria modelo calculando a melhor reta = OrdinaryListSquares(OLS)
modelo = sm.OLS(y1, X1).fit()
modelo.summary()

# p > NS
# curbweight = 0.222 > 0.05
X2 = X1.drop(['curbweight'], axis=1)
modelo = sm.OLS(y1, X2).fit()
modelo.summary()


# separar dados de testes e dados de treinamento
X_train, X_test, y_train, y_test = train_test_split(
    X2, y1, test_size=0.2, random_state=0)

# regressao linear multipla
regressor = LinearRegression()
regressor.fit(X_train, y_train) # Treina o regressor
y_pred = regressor.predict(X_test) # Faz predição

# comparando valores previstos X reais
compara_ys = np.concatenate(
    (np.around(y_pred, 1).reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1)

# Acurácia modelo2
# *fueltype             tipo de combustível
# *carwidth             largura de carro
# *enginesize           tamanho do motor
# *horsepower           cavalo-vapor
mse = round((mean_squared_error(y_test, y_pred))/100, 2)
rmse = round((sqrt(mse))/100, 2)





