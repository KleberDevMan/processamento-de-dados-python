import numpy as np
import pandas as pd

dataSetIris = pd.read_csv('dados corona brasil (corona vírus)/brazil_covid19.csv')
 
# variáveis independentes (matrix; são as caracteristicas; atributos usados para chegar a um resultado)
X = dataSetIris.iloc[:,[0,2,5]].values
# variáveis dependentes (vetor; o resultado; depende das variáveis independentes)
y = dataSetIris.iloc[:,6].values

# CODIFICANDO DADOS CATEGORICOS/STRING
# Tocantins:    0 0 1
# Maranhao:     0 1 0
    # ...
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
# code_X = ColumnTransformer([('codificar_X', OneHotEncoder(), [0])], remainder='passthrough')
code_X = ColumnTransformer([('codificar_X', OrdinalEncoder(), [0,1])], remainder='passthrough')
X = np.array(code_X.fit_transform(X))

# # TESTANDO A NORMALIDADE DOS DADOS
# import scipy.stats as stats

# p = [1,2,3]
# a = ['','','']
# for i in range(3):
#     a[i], p[i] = stats.shapiro(X[:,i])

# i = 0
# for x in p:
#     if x >= 0.05:
#         print("coluna {}º NÃO-NORMAL. valor teste: ({})".format(i, x))
#     else:
#         print("coluna {}º NORMAL. valor teste: ({})".format(i, x))
#     i+=1

# # PADRONIZANDO OS DADOS
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# # SEPARAR A BASE_DADOS(train; treinamento) DOS DADOS_TEST(testes)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)