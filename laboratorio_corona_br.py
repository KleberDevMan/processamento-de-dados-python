import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dados corona brasil (corona vírus)/brazil_covid19_v2.csv')

print(data.describe())
print(data.corr())
print(data.cov())

# 2. DANDO UMA OLHADA NOS DADOS
d = data['cases']
import functions as func

#histograma
func.histograma_bins(d, 'Casos')

#densidade
func.grafico_densidade(d)

#casos confirmados e mortes ao longo do tempo
func.mortes_casos_ao_longo_tempo(data)

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(data.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()