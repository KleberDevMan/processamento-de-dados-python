
### Regressão Linear Simples
### SVR

### ATIVIDADE:
### Aplicar Regressão Polinomial de grau 2 e 3
### Comparar os resultados -> ver tabela no final do código

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

### Numero de Casos
### Numero de Mortes

# tratamentos
df_countries_cases       = df_covid19.copy().drop(['Lat','Long_','Last_Update'],axis =1)
df_countries_cases.index = df_countries_cases["Country_Region"]
df_countries_cases       = df_countries_cases.drop(['Country_Region'],axis=1)

df_conf_deaths = df_countries_cases.iloc[:,0:2]
df_conf_deaths.corr().style.background_gradient(cmap='Reds')

# Matrix de caracteristicas -> variaveis independentes
X = df_conf_deaths.iloc[:, :-1].values
# Vector de variaveis dependentes
y = df_conf_deaths.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

### RLS ###
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred_rls = regressor.predict(X_test)

# prevendo o número de mortes
# quando temos 220.000 casos confirmados
regressor.predict([[220000]])

# Visualising the Linear Regression results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Regressão Linear Simples - Base Treinamento')
plt.xlabel('Casos')
plt.ylabel('Mortes')
plt.show()


### SVR ###
# convertendo para matriz (nX1) Obs.: (linhasXcolunas)
y_train_svr = y_train.reshape(len(y_train),1)
y_test_svr = y_test.reshape(len(y_test),1)

# Escalando os dados (procedimento necessário para realizar o SVR)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train) # já está no padrão: (nX1) Obs.: (linhasXcolunas)
y_train_svr = sc_y.fit_transform(y_train_svr)

X_test_svr = sc_X.fit_transform(X_test)
# realizar correção* (pois sem a correcao vai dar 7211 mortes, quando 220.000 casos confirmados )
y_test_svr = sc_y.fit_transform(y_test_svr) 

# criando o modelo de regressao SVR
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'poly')
regressor_svr.fit(X_train_svr, y_train_svr)

# prevendo o número de mortes
# quando temos 220.000 casos confirmados
sc_y.inverse_transform(regressor_svr.predict(sc_X.transform([[220000]]))) # result: 11.130 mortes


## Metricas de qualidade
# RLS
from sklearn.metrics import mean_squared_error
mse_rls = mean_squared_error(y_test, regressor.predict(X_test))
rmse_rls = round(np.sqrt(mse_rls),2)

res_rls = {"MSE_RLS": mse_rls, "RMSE_RLS": rmse_rls}
print('RMSE_RLS: ', res_rls['RMSE_RLS'])

# SVR
mse_svr = mean_squared_error(X_test_svr, sc_y.inverse_transform(regressor_svr.predict(X_test_svr)))
rmse_svr = round(np.sqrt(mse_svr),2)

res_svr = {"MSE_SVR": mse_svr, "RMSE_SVR": rmse_svr}
print('RMSE_SVR: ', res_svr['RMSE_SVR'])

### RP ###
# Modelo de predição usando Regressão Polinomial do 2° grau
from sklearn.preprocessing import PolynomialFeatures
reg_poli = PolynomialFeatures(degree=3) # regressao polinomial grau 2

X_poli = reg_poli.fit_transform(X_train)

# y = b0 + b1 * x1 + b2 * x1^2 + b3 * x1^3
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poli, y_train)

# Visualizando os resultados da Regressão Polinomial
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lin_reg2.predict(X_poli), color = 'blue')
plt.title('Regressão Linear Polinomial - Dados de treinamento')
plt.xlabel('Casos')
plt.ylabel('Mortes')
plt.show()

## Metricas de qualidade
# RP
mse_rp = mean_squared_error(y_train, lin_reg2.predict(X_poli))
rmse_rp = round(np.sqrt(mse_rp),2)

res_rp = {"MSE_RP": mse_rp, "RMSE_RP": rmse_rp}
print("RMSE_RP: ",res_rp["RMSE_RP"])


### Criar a regressão polinomial de grau 2 e 3
### e comparar os resultados conforme tabela abaixo
### RMSE
# | RLS     | SVR-rbf | SVR-linear | SVR-poly | Poly grau=2 | Poly grau = 3
# | 3161.03 | 2565.96 | 5725.78    | 1460.06  |             |


# Metricas de qualidade dia 31/05/2020 (obs.: com a correção*)
# | RLS     | SVR-rbf | SVR-linear | SVR-poly | Poly grau=2 | Poly grau = 3
# | 3472.52 | 3317.65 | 5958.37    | 1414.52  | 3143.66     | 3019.72
