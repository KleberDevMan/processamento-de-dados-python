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



    