import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

from resources import BASE_PATH

def getScores(scores,sents):
    ss = []
    for ii in range(len(sents)):
        sent = sents[ii]
        if len(word_tokenize(sent)) <= 10:
            continue
        elif '-----' in sent:
            continue
        else:
            ss.append(scores[ii])
    return ss


def mergeEntries(name):
    data = pd.read_csv('{}/Discussion&Response/{}.csv'.format(BASE_PATH,name))
    new_data = data.loc[:,['PID','Reviewer']]
    ff = open('../TokenisedResponses/{}.index'.format(name),'r')
    idx = [int(ll) for ll in ff.readlines()]
    ff = open('{}.pol_scores'.format(name),'r')
    scores = [float(ll) for ll in ff.readlines()]
    ff = open('../TokenisedResponses/{}.sent'.format(name),'r')
    sents = [ll for ll in ff.readlines()]
    ff.close()
    assert len(idx) == data.shape[0]

    pointer = 0
    mins = []
    maxs = []
    means = []
    medians = []
    stds = []
    for ii in idx:
        if ii != 0:
            ss = getScores(scores[pointer:pointer+ii],sents[pointer:pointer+ii])
            if len(ss) > 0:
                mins.append(np.min(ss))
                maxs.append(np.max(ss))
                means.append(np.mean(ss))
                medians.append(np.median(ss))
                stds.append(np.std(ss))
                pointer += ii
                continue
        mins.append('')
        maxs.append('')
        means.append('')
        medians.append('')
        stds.append('')

    new_data['min'] = mins
    new_data['max'] = maxs
    new_data['mean'] = means
    new_data['median'] = medians
    new_data['std'] = stds

    new_data.to_csv('{}_politeness.csv'.format(name))
    return new_data

if __name__ == '__main__':
    names = ['Increase','Decrease','NoChange']

    all_df = None
    for nn in names:
        data = mergeEntries(nn)
        if all_df is None:
            all_df = data
        else:
            all_df = all_df.append(data,sort=False)

    all_df.to_csv('all_politeness.csv')

