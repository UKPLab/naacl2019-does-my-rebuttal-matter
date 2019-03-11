import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

def getStats(name):
    ff = open('{}.pol_scores'.format(name),'r')
    scores = []
    for line in ff.readlines():
        scores.append(float(line))
    ff.close()

    print('\n=== Politeness Scores in {} === '.format(name))
    print('max : {}'.format(np.max(scores)))
    print('min : {}'.format(np.min(scores)))
    print('mean : {}'.format(np.mean(scores)))
    print('median : {}'.format(np.median(scores)))
    print('std. dev. : {}'.format(np.std(scores)))


def getMinMaxMean(name):
    ff = open('{}.pol_scores'.format(name),'r')
    scores = []
    for line in ff.readlines():
        scores.append(float(line))
    ff.close()

    ff = open('../TokenisedResponses/{}.index'.format(name),'r')
    idx = [int(ll) for ll in ff.readlines()]
    ff.close()

    mins = []
    maxs = []
    means = []
    medians = []

    pointer = 0
    for ii in idx:
        if ii != 0:
            ss = scores[pointer:pointer+ii]
            mins.append(np.min(ss))
            maxs.append(np.max(ss))
            means.append(np.mean(ss))
            medians.append(np.median(ss))
            pointer += ii

    return mins, maxs, means, medians


def plotDist(names):
    all_scores = OrderedDict()

    minv = 9999
    maxv = -9999
    for nn in names:
        mins, maxs, means, medians = getMinMaxMean(nn)
        scores = maxs[:]
        if np.min(scores) < minv: minv = np.min(scores)
        if np.max(scores) > maxv: maxv = np.max(scores)
        all_scores[nn] = scores

    label_names = {'Increase':'iResp', 'Decrease':'dResp', 'NoChange':'kResp'}
    line_type = {'Increase':'solid', 'Decrease':'dashed', 'NoChange':'dotted'}
    colors = {'Increase':'red', 'Decrease':'blue', 'NoChange':'green'}
    bins = np.linspace(0,10,50)
    for nn in names:
        norms = [(v-minv)*10./(maxv-minv) for v in all_scores[nn]]
        all_scores[nn] = norms[:]
        sns.kdeplot(all_scores[nn],alpha=1,label=label_names[nn],ls=line_type[nn],color=colors[nn])
        #plt.hist(all_scores[nn],bins,alpha=0.3,label=nn,density=True)

    font_size = 18
    plt.rc('font',size=font_size)
    plt.xlabel('plt_max scores',fontsize=font_size)
    plt.ylabel('density of probability',fontsize=font_size)
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == '__main__':
    names = ['Increase','Decrease','NoChange']
    plotDist(names)
    #for nn in names:
        #getStats(nn)


