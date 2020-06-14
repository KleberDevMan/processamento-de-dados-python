# importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importando a base de dados
dataset = pd.read_csv('compra.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Escalando os dados - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Treinamento usando k-nn
from sklearn.neighbors import KNeighborsClassifier
# minkowski com p=2 significa distância euclidiana (default) 
classificador = KNeighborsClassifier(n_neighbors=5, 
    metric = 'minkowski', p = 2)
classificador.fit(X_train, y_train)

# Predizendo um novo resultado
idade = 50
salario = 80000
classificador.predict(sc.transform([[idade, salario]]))
classificador.predict_proba(sc.transform([[idade, salario]]))

# Predizendo dados de teste
y_pred = classificador.predict(X_test)
compara_ys = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Matriz de confusão:
#   exibição gráfica do quanto eu acertei
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import plot_confusion_matrix
plt.rcParams['figure.figsize'] = [12, 12]
plot_confusion_matrix(classificador, X_test, y_test, normalize='true')

# Medindo acurácia
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
