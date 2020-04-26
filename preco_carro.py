# Atividade de Regressao Linear Simples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# caracteristicas e preco de um carro
data = pd.read_csv('carros_data.csv')

# grafico de calor para verficar variáveis correlacionadas
f, ax = plt.subplots(figsize=(24, 21))
sns.heatmap(data.corr(), annot=True, linewidths=10.0, ax=ax)

# correlacao do preco com o tamanho do motor
data['price'].corr(data['enginesize'])

# variaveis independentes
TMOTOR = data.iloc[:, 16].values
# variavel dependente
preco = data.iloc[:, -1].values

# separar dados de testes e dados de treinamento
from sklearn.model_selection import train_test_split
TMOTOR_train, TMOTOR_test, preco_train, preco_test = train_test_split(TMOTOR, preco, test_size = 0.2, train_size = 0.8)

# regressao linear simples
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

## Preparando o regressor
# encotra a melhor predicao/regressao (melhor reta)
regressor.fit(TMOTOR_train.reshape(-1, 1), preco_train)
regressor.coef_
regressor.intercept_
## Dúvida: Porque a cada execução o regressor gera variáveis novas? É por causa da separação dos dados de teste e treinamento.
# Coeficiente: b1 = 161.45
# Constante: b0 = -7337.47 --> significa que se o carro vier sem motor o dono deve ser ressarcido em aprox. U$ 7337.47
# preco = b0 + b1 * tam_motor
# preco = 161.45 + -7337.47 * tam_motor

## Fazendo a predição do valor de alguns carros usando dados de teste
preco_pred = regressor.predict(TMOTOR_test.reshape(-1, 1))

# comparando valores previstos com os valores reais
preco_pred_obs = [preco_pred, preco_test]

## Visualizacao dos dados Gráfico Scatter
# predicao dos precos dos carros nos dados de treinamento
plt.rcParams['figure.figsize'] = [12, 12]
plt.scatter(TMOTOR_train, preco_train, color = 'red')
plt.plot(TMOTOR_train, regressor.predict(TMOTOR_train.reshape(-1, 1)), color = 'black')
plt.title('Tamanho do Motor vs. Preço - Treinamento')
plt.xlabel('Tamanho do motor')
plt.ylabel('Preço')
plt.show()

# predicao do preço dos carros nos dados de teste
plt.rcParams['figure.figsize'] = [12, 12]
plt.scatter(TMOTOR_test, preco_test, color = 'red')
plt.plot(TMOTOR_test, regressor.predict(TMOTOR_test.reshape(-1, 1)), color = 'black')
plt.title('Tamanho do Motor vs. Preço - Teste')
plt.xlabel('Tamanho do motor')
plt.ylabel('Preço')
plt.show()





