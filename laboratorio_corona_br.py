import numpy as np
import pandas as pd

data = pd.read_csv('dados corona brasil (corona vírus)/brazil_covid19_v2.csv')

print(data.describe())
print(data.corr())
print(data.cov())


# 1. PRE-PŔOCESSANDO OS DADOS
# # variáveis independentes (caracteristicas; atributos)
# X = data.iloc[:,[2,3]].values
# # variáveis dependentes (resultado; depende dos attr)
# y = data.iloc[:,4].values

# # CODIFICANDO DADOS CATEGORICOS/STRING
# # Tocantins:    0 0 1
# # Maranhao:     0 1 0
# from sklearn.compose import ColumnTransformer
# import category_encoders as ce
# col_X = ColumnTransformer([('code_X', ce.OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(col_X.fit_transform(X))

# # # PADRONIZANDO OS DADOS
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)

# # # SEPARAR A BASE_DADOS(train; treinamento) DOS DADOS_TEST(testes)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)




# 2. DANDO UMA OLHADA NOS DADOS
# d = data['cases']
# import functions as func

#histograma
# func.histograma_bins(d, 'Mortes')

#densidade
# func.grafico_densidade(d)

#casos confirmados e mortes ao longo do tempo
# func.mortes_casos_ao_longo_tempo(data)

# COVARIANCIA(indício de correlação): variabilidade conjunta entre variaveis
#   positiva: se uma crescer, outra cresce
#   negativa: se uma crescer, outra diminui
#   magnitude: quanto mais proximo de zero, menor o efeito de uma sobre a outra

# COVARIANCIA ENTRE TODAS


# CORRELACAO: Mostra a relação entre um par de variáveis
#   magnitude: 
#       - quanto mais proximo de 1, mais forte é a correlacao positiva
#       - quanto mais proximo de -1, mais forte é a correlacao negativa

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(dpi=100)
# plt.title('Correlation Analysis')
# sns.heatmap(data.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
# plt.xticks(rotation=60)
# plt.yticks(rotation = 60)
# plt.show()