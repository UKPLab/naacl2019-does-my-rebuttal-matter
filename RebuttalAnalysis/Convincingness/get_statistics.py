import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

def getStats(name):
    data = pd.read_csv('{}_consent.cvc'.format(name))
    ss = data['Cvc-min'] #cvc-min seems giving the most sensible ranking
    scores = [s for s in ss if s<0]

    print('\n=== Politeness Scores in {} === '.format(name))
    print('max : {}'.format(np.max(scores)))
    print('min : {}'.format(np.min(scores)))
    print('mean : {}'.format(np.mean(scores)))
    print('median : {}'.format(np.median(scores)))
    print('std. dev. : {}'.format(np.std(scores)))

def plotDist(names):
    all_scores = OrderedDict()

    minv = 9999
    maxv = -9999
    for nn in names:
        data = pd.read_csv('{}_cvc.csv'.format(nn))
        ss = data['min'] #cvc-min seems giving the most sensible ranking
        scores = [-s for s in ss if s<0]
        all_scores[nn] = scores
        if np.min(scores) < minv: minv = np.min(scores)
        if np.max(scores) > maxv: maxv = np.max(scores)

    bins = np.linspace(0,10,50)
    label_names = {'Increase':'iResp', 'Decrease':'dResp', 'NoChange':'kResp'}
    line_type = {'Increase':'solid', 'Decrease':'dashed', 'NoChange':'dotted'}
    colors = {'Increase':'red', 'Decrease':'blue', 'NoChange':'green'}
    for nn in names:
        norms = [(v-minv)*10./(maxv-minv) for v in all_scores[nn]]
        all_scores[nn] = norms[:]
        sns.kdeplot(all_scores[nn],alpha=1,label=label_names[nn],ls=line_type[nn],color=colors[nn])
        #plt.hist(all_scores[nn],bins,alpha=0.3,label=nn,density=True)

    font_size = 18
    plt.rc('font',size=font_size)
    plt.xlabel('cvc_min scores',fontsize=font_size)
    plt.ylabel('density of probability',fontsize=font_size)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    names = ['Increase','Decrease','NoChange']
    plotDist(names)
    #for nn in names:
    #    getStats(nn)



