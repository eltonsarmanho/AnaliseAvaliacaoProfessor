import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from empiricaldist import Cdf
from empiricaldist import Pmf
from matplotlib import cm
from matplotlib.colors import ListedColormap
import  empiricaldist
import scipy
import matplotlib.pyplot as plt


def loadData():
    try:
        dataset = pd.read_csv("./Dataset/ED1_Turma_2022.csv",low_memory=False);
        dataset.drop(columns=['Carimbo de data/hora'],inplace=True)

        print(dataset.describe())
        print(dataset.info())
        return dataset;
    except:
         print("Oops!", sys.exc_info()[0], "occurred.");

def barPlot(dataset):
    feature = 'Nível de esforço [Seu nível de dedicação ao curso]'
    dataset_subset = dataset[dataset[feature] != 'Fraco'].copy()
    dataset_subset.dropna(subset=[feature],
                          inplace=True)
    sns.countplot(x=feature, data=dataset_subset)
    plt.show()

def piePlot(dataset):

    features_1 = [
               "[O professor foi um palestrante eficiente]",
               "[As apresentações foram claras e organizadas]",
               "[O professor usou bem os recursos e as ferramentas computacionais]",
               "[O professor foi acessível e prestativo]",
               ]
    #title = "Habilidade e receptividade do Professor "
    features_2 = ['Nível de esforço dos alunos',
                "[Os objetivos foram claros]",
                "[O conteúdo da disciplina foi organizada e bem planejada]",
                "[A carga da disciplina foi apropriada]" ]
    title = "Disciplina"
    fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(4, 2), dpi=100)
    plt.suptitle(title)
    axe = axes.ravel()
    for i, feature in enumerate(features_2):
        dataset_copy = dataset.copy()
        dataset_copy.dropna(subset=[feature],
                            inplace=True)

        df = dataset_copy.groupby([feature]).agg({feature: ["count"]})
        df.reset_index(inplace=True)
        df.columns = [feature, 'Quantidade']
        axe[i].pie(df['Quantidade'], labels=df[feature], startangle=90, autopct='%1.0f%%', textprops={'fontsize': 14})
        axe[i].set_title(feature + ' em % ',fontsize=10, bbox={'facecolor': '0.8', 'pad': 5})

    plt.show()


def cdfPlot(dataset):
    feature = 'Nível de esforço [Seu nível de dedicação ao curso]'
    labels = {"Excelente": 5,"Muito bom": 4,
                                "Satisfatório": 2,"Moderado": 1,"Fraco":0}
    dataset[feature].replace(labels, inplace=True)
    #dataset["feature"].replace({"Concordo plenamente": 5, "Concordo": 4,
    #                            "Não sei": 3, "Discordo": 2, "Discordo totalmente": 1}, inplace=True)


    dataset.dropna(subset=[feature],
                          inplace=True)

    for value in [feature]:
        # sort data
        data_sorted = np.sort(dataset[value])
        # calculate CDF values
        norm_cdf = scipy.stats.norm.cdf(data_sorted)
        # plot CDF
        #plt.plot(data_sorted, norm_cdf,label=value,alpha=0.4)

    plt.legend()
    # plot CDF
    plt.axvline(x=2)

    plt.xticks(ticks=list(labels.values()),labels=list(labels.keys()))

    plt.xlabel('Quantidade')
    plt.ylabel('CDF')
    plt.show()

if __name__ == '__main__':
    dataset = loadData()
    piePlot(dataset)