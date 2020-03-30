import numpy as np
import pandas as pd

dataSetIris = pd.read_csv('dados bodyfat (gordura corporal)/500_Person_Gender_Height_Weight_Index.csv')
 
# variáveis independentes (matrix; são as caracteristicas; atributos usados para chegar a um resultado)
X = dataSetIris.iloc[:,1::].values
# variáveis dependentes (vetor; o resultado; depende das variáveis independentes)
y = dataSetIris.iloc[:,0].values

# CODIFICANDO DADOS CATEGORICOS/STRING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
# codifica variavel independente
code_y = LabelEncoder()
y = code_y.fit_transform(y)

# # TESTANDO A NORMALIDADE DOS DADOS
import scipy.stats as stats

p = [1,2,3]
for i in range(3):
    a, p[i] = stats.shapiro(X[:,i])

print("Teste de Shapiro-Wilk {}".format(a))

i = 0
for x in p:
    if x >= 0.05:
        print("coluna {}º NÃO-NORMAL. valor teste: ({})".format(i, x))
    else:
        print("coluna {}º NORMAL. valor teste: ({})".format(i, x))
    i+=1

# # PADRONIZANDO OS DADOS
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# # SEPARAR A BASE_DADOS(train; treinamento) DOS DADOS_TEST(testes)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)