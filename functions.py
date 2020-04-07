import matplotlib.pyplot as plt

def histograma_bins(d, label):
    for i, largura_bin in enumerate([1, 5, 10, 15]):
        # cria o plot
        ax = plt.subplot(2, 2, i + 1)

        # desenha o grafico
        ax.hist(d, bins = int(180/largura_bin),
            color = 'blue', edgecolor = 'black')

        # titulos e labels
        ax.set_title('Bin = %d' % largura_bin,
            size = 30)
        ax.set_xlabel(label)
        ax.set_ylabel('Frequencia')

import seaborn as sns
def grafico_densidade(d):
        sns.distplot(d, hist=True, kde=True,
            bins=int(180/5),
            color = 'blue',
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth':6})


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Layout
def mortes_casos_ao_longo_tempo(data):
    data_over_time = data[['cases','deaths']].groupby(data['date']).sum().sort_values(by = 'cases', ascending=True)

    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)'
        , plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = make_subplots(rows=2, cols=1
                        , subplot_titles=('Casos confirmados', 'Mortes'))

    fig.append_trace(go.Line(name='Confirmados'
                            , x = data_over_time.index
                            , y = data_over_time['cases'])
                            , row=1, col=1)

    fig.append_trace(go.Line(name='Mortes'
                            , x = data_over_time.index
                            , y = data_over_time['deaths'])
                            , row=2, col=1)

    fig['layout'].update(layout)

    fig.show()


    