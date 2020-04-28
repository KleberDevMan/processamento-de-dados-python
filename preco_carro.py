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
    X, y, test_size=0.2, train_size=0.8)

# regressao linear multipla
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Preparando o regressor
# encotra a melhor predicao/regressao (melhor reta)
regressor.fit(X_train, y_train)
regressor.coef_
regressor.intercept_
# Dúvida: Porque a cada execução o regressor gera variáveis novas? É por causa da separação dos dados de teste e treinamento.
# Coeficiente: b1 = 161.45
# Constante: b0 = -7337.47 --> significa que se o carro vier sem motor o dono deve ser ressarcido em aprox. U$ 7337.47
# y = b0 + b1 * tam_motor
# y = 161.45 + -7337.47 * tam_motor

# Fazendo a predição do valor de alguns carros usando dados de teste
y_pred = regressor.predict(X_test)

# comparando valores previstos com os valores reais
compara_ys = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)