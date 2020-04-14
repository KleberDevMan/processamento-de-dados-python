import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('mushrooms.csv')

# pre-visualizacao
data
data.shape

# verifico os valores nulos
data.isnull().sum()

# Codifica as colunas LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
 
data.head()

# cogumelos venenosos e comestiveis
data.groupby('class').size()

# Correlacao
plt.figure(figsize=(25, 15))
sns.heatmap(data.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True)

# Grafico boxplot inline
# usado para 
plt.figure(figsize=(60, 6))
sns.boxplot(x="variable", y="value", data=pd.melt(data))

# Regressao linear simples
sns.regplot(x=data['veil-color'], y=data['gill-attachment'])

# Codifica as colunas HotEncoder
df_one_hot = pd.concat([data.iloc[:,0], pd.get_dummies(data.iloc[:,1:23])], axis=1)
features = df_one_hot.iloc[:,1:118]
label = df_one_hot.iloc[:,0] 
label = le.fit_transform(label)
