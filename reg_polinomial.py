
# importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importando a base de dados
dataset = pd.read_csv('salario_por_posicao.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# nós não vamos separar em treinamento e
# teste

# trainamento do modelo de regressao linear
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# treinar com toda a base de dados
#   o modelo de regressão polinomial

from sklearn.preprocessing import PolynomialFeatures
reg_poli = PolynomialFeatures(degree=2) # regressao polinomial grau 2

X_poli = reg_poli.fit_transform(X)

# y = b0 + b1 * x1 + b2 * x1^2 + b3 * x1^3
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poli, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Regressão Linear Simples')
plt.xlabel('Posição')
plt.ylabel('Salário')
plt.show()

# Visualizando os resultados da Regressão Polinomial
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_poli), color = 'blue')
plt.title('Regressão Linear Polinomial')
plt.xlabel('Posição')
plt.ylabel('Salário')
plt.show()

# reg simples
lin_reg.predict([[6.5]])

# reg polinomial
lin_reg2.predict(reg_poli.fit_transform([[6.5]]))

# metricas de qualidade
# RLS
from sklearn.metrics import mean_squared_error
mse_rls = mean_squared_error(y, lin_reg.predict(X))
rmse_rls = round(np.sqrt(mse_rls),2)

res_rls = {"MSE_RLS": mse_rls, "RMSE_RLS": rmse_rls}
print(res_rls)

# RP
mse_rp = mean_squared_error(y, lin_reg2.predict(X_poli))
rmse_rp = round(np.sqrt(mse_rp),2)

res_rp = {"MSE_RP": mse_rp, "RMSE_RP": rmse_rp}
print(res_rp)


##### SVR #####
y_mat = y.reshape(len(y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_sc = sc_X.fit_transform(X)
y_sc = sc_y.fit_transform(y_mat)

# Treinamento na base de dados toda
# # https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200
from sklearn.svm import SVR
reg_svr = SVR(kernel='rbf')
reg_svr.fit(X_sc, y_sc)

sc_y.inverse_transform(reg_svr.predict(sc_X.transform([[6.5]])))

# SVR
mse_svr = mean_squared_error(y_sc, sc_y.inverse_transform(reg_svr.predict(X_sc)))
rmse_svr = round(np.sqrt(mse_svr),2)

res_svr = {"MSE_SVR": mse_svr, "RMSE_SVR": rmse_svr}
print(res_svr)

# Visualizacao
plt.scatter(sc_X.inverse_transform(X_sc), sc_y.inverse_transform(y_sc), color = 'red')
plt.plot(sc_X.inverse_transform(X_sc), sc_y.inverse_transform(reg_svr.predict(X_sc)), color = 'blue')
plt.title('SVR')
plt.xlabel('Posição')
plt.ylabel('Salário')
plt.show()