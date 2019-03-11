import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

def getStats(name):
    data = pd.read_csv('{}.sim'.format(name))
    embd = data['EmbdSimilarity'].values
    rouge1 = data['Rouge1Similarity'].values
    rouge2 = data['Rouge2Similarity'].values

    scores = {'Embd':embd, 'Rouge1':rouge1, 'Rouge2':rouge2}

    print('\n=== Review-Response Similarity in {} === '.format(name))
    for stype in scores:
        print('---Similarity in terms of {}---'.format(stype))
        print('max : {}'.format(np.max(scores[stype])))
        print('min : {}'.format(np.min(scores[stype])))
        print('mean : {}'.format(np.mean(scores[stype])))
        print('median : {}'.format(np.median(scores[stype])))
        print('std. dev. : {}'.format(np.std(scores[stype])))

def plotDist(names):
    all_scores = OrderedDict()

    minv = 9999
    maxv = -9999
    for nn in names:
        data = pd.read_csv('{}.sim'.format(nn))
        ss = data['EmbdSimilarity'].values #cvc-min seems giving the most sensible ranking
        scores = [ele for ele in ss if ele>0]
        all_scores[nn] = scores
        if np.min(scores) < minv: minv = np.min(scores)
        if np.max(scores) > maxv: maxv = np.max(scores)

    bins = np.linspace(0,10,50)
    for nn in names:
        norms = [(v-minv)*10./(maxv-minv) for v in all_scores[nn]]
        all_scores[nn] = norms[:]
        sns.kdeplot(all_scores[nn],alpha=0.3,label=nn)
        #plt.hist(all_scores[nn],bins,alpha=0.3,label=nn,density=True)

    plt.legend()
    plt.show()


def getCorrelation():
    data = pd.read_csv('all.sim')
    embd = data['EmbdSimilarity'].values
    rouge1 = data['Rouge1Similarity'].values
    rouge2 = data['Rouge2Similarity'].values

    print('\n=====CORRELATIONS=====')
    print('pearson correlation between embd and rouge1: {}'.format(pearsonr(embd,rouge1)))
    print('pearson correlation between embd and rouge2: {}'.format(pearsonr(embd,rouge2)))
    print('pearson correlation between rouge1 and rouge2: {}'.format(pearsonr(rouge1,rouge2)))
    print('=====CORRELATIONS=====\n\n')

if __name__ == '__main__':
    #getCorrelation()

    names = ['Increase','Decrease','NoChange']
    #for nn in names:
    #   getStats(nn)
    plotDist(names)
