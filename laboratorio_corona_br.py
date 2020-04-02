import numpy as np
import pandas as pd

dataSetIris = pd.read_csv('dados corona brasil (corona vírus)/brazil_covid19.csv')
 
# variáveis independentes (caracteristicas; atributos)
X = dataSetIris.iloc[:,[2,5]].values
# variáveis dependentes (resultado; depende dos attr)
y = dataSetIris.iloc[:,6].values

# CODIFICANDO DADOS CATEGORICOS/STRING
# Tocantins:    0 0 1
# Maranhao:     0 1 0
from sklearn.compose import ColumnTransformer
import category_encoders as ce
col_X = ColumnTransformer([('code_X', ce.OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(col_X.fit_transform(X))

# # PADRONIZANDO OS DADOS
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# # SEPARAR A BASE_DADOS(train; treinamento) DOS DADOS_TEST(testes)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)