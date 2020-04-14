import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions as func

data = pd.read_csv('diabetes.csv')

# pre-visualizacao
data
data.shape

# verifico os valores nulos
data.isnull().sum()

# diabeticos e não-diabeticos
data.groupby('Outcome').size()

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
sns.regplot(x=data['Pregnancies'], y=data['Age'])

#histograma
func.histograma_bins(data['Pregnancies'], 'Pregnancies')

#densidade
func.grafico_densidade(data['Pregnancies'])