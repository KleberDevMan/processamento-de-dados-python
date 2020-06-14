import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Carregar base de dados
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

#  Treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# plt.figure(figsize=(20,4))
# for index, (imagem, categoria) in enumerate(zip(X_train[0:5], y_train[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(imagem, (28,28)), cmap=plt.cm.gray)
#     plt.title('Treinamento: ' + str(categoria), fontsize = 20)

# Classificador
from sklearn.svm import SVC
# Como escolher o kerner:
#   Dados dispostos de forma rbf, kernel default do SVC
#   Dados dispostos de forma polinomial, curva? => poly
#   Dados dispostos de forma linear, reta? => linear
#   Dados dispostos de forma radial, grupos? => radial

# classificador = SVC(kernel = 'rbf', random_state = 0)
classificador = SVC(kernel = 'poly', random_state = 0)
classificador.fit(X_train, y_train)

# plt.imshow(np.reshape(X_test[5], (28,28)))
# classificador.predict(X_test[5].reshape(1,-1))

# Fazendo as predições de todos os dados
import time
tini=time.time()
y_pred = classificador.predict(X_test)
tfim=time.time()
print('todas predições: ', round(tfim - tini, 2), 's')

# Matrix de confusão e acurácia
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# cm_df = pd.DataFrame(cm, index = ['0','1','2','3','4','5','6','7','8','9'],
#                   columns = ['0','1','2','3','4','5','6','7','8','9'])

from sklearn.metrics import plot_confusion_matrix

plt.rcParams['figure.figsize'] = [12, 12]
plot_confusion_matrix(classificador, X_test, y_test)
plot_confusion_matrix(classificador, X_test, y_test, normalize='true')
plt.show()


# Acurácia
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)


# Imagens classificadas incorretamente
# index = 0
# falso_positivo = []
# for categoria, predito in zip(y_test, y_pred):
#     if categoria != predito: 
#         falso_positivo.append(index)
#     index +=1

# plt.figure(figsize=(20,4))
# for plotIndex, erroIndex in enumerate(falso_positivo[0:5]):
#     plt.subplot(1, 5, plotIndex + 1)
#     plt.imshow(np.reshape(X_test[erroIndex], (28,28)), cmap=plt.cm.gray)
#     plt.title('Predito: {}, Real: {}'.format(y_pred[erroIndex], y_test[erroIndex]), fontsize = 15)