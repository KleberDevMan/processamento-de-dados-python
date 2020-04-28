# Atividade de Regressao Linear Simples
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# caracteristicas e y de um carro
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
X = data.iloc[:, [3, 13, 16]]
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

# Constante: b0 = -14372.43 --> significa que se o carro tiver zero nas variaveis independentes o dono deve ser ressarcido em aprox. U$ 14372.43
# Coeficiente: b1 = 331.59
# Coeficiente: b2 = 5.38
# Coeficiente: b3 = 106.23
# preco = b0 + b1 * fueltype + b2 * curbweight + b3 * enginesize
# preco = -13800.20 + 331.59 * fueltype + 5.38 * curbweight + 106.23 * enginesize 

# Fazendo a predição do valor dos carros dos dados de teste
y_pred = regressor.predict(X_test)

# comparando valores previstos com os valores reais
compara_ys = np.concatenate(
    ( np.around(y_pred,1).reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
