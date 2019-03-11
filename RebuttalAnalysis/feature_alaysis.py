import numpy as np
import scipy.stats as sta
from nltk.tokenize import word_tokenize
import random

def getFeatureScore(name,feature):
    ff = open('./TokenisedResponses/{}.index'.format(name),'r')
    lines = ff.readlines()
    index_list = [int(ll) for ll in lines]
    ff.close()

    if feature == 'Specificity':
        ff = open('{}/{}.spec'.format(feature,name), 'r')
    elif feature == 'Politeness':
        ff = open('{}/{}.pol_scores'.format(feature,name), 'r')
    else:
        print('Invalid feature name {}! Exit!'.format(feature))
        exit(11)
    lines = ff.readlines()
    score_list = [float(ll) for ll in lines]
    ff.close()

    avg_score_list = []
    max_score_list = []
    med_score_list = []

    pointer = 0
    for ii in range(len(index_list)):
        sent_num = index_list[ii]
        scores = score_list[pointer:pointer+sent_num]
        if len(scores) > 0:
            avg_score_list.append(np.mean(scores))
            max_score_list.append(max(scores))
            med_score_list.append(np.median(scores))
        else:
            avg_score_list.append(0)
            max_score_list.append(0)
            med_score_list.append(0)
        pointer += sent_num

    print('\n==={} ({} responses)==='.format(name,len(index_list)))
    print('avg specificity scores: {}, std err {}'.format(np.mean(avg_score_list),sta.sem(avg_score_list)))
    print('max specificity scores: {}, std err {}'.format(np.mean(max_score_list),sta.sem(max_score_list)))
    print('median specificity scores: {}, std err {}'.format(np.mean(med_score_list),sta.sem(med_score_list)))

    return index_list, score_list


def readArguments(path):
    ff = open(path,'r')
    arguments = ff.readlines()
    ff.close()
    return arguments


def getTopBottom(feature):
    names = ['Increase','Decrease','NoChange']
    scores = []
    arguments = []
    for nn in names:
        _,ss = getFeatureScore(nn,feature)
        scores.extend(ss)
        aa = readArguments('TokenisedResponses/{}.sent'.format(nn))
        arguments.extend(aa)

    assert len(scores) == len(arguments)

    num = 10
    sorted_list = sorted(scores,reverse=True)

    last = 12.3456
    cnt = 0
    print('===top {}===\nscore\targ'.format(num))
    for ss in sorted_list:
        if ss == last:
            continue
        else:
            idx = scores.index(ss)
            arg = arguments[idx]
            if len(word_tokenize(arg)) <= 10:
                continue
            elif '-----' in arg:
                continue
            print('{}\t{}'.format(ss,arg))
            last = ss
            cnt += 1
            if cnt >= num:
                break


    last = 12.3456
    cnt = 0
    print('\n===bottom {}===\nscore\targ'.format(num))
    for ii in range(len(sorted_list)):
        ss = sorted_list[len(sorted_list)-ii-1]
        if ss == last:
            continue
        else:
            idx = scores.index(ss)
            arg = arguments[idx]
            if len(word_tokenize(arg)) <= 10:
                continue
            elif '-----' in arg:
                continue
            print('{}\t{}'.format(ss,arg))
            last = ss
            cnt += 1
            if cnt >= num:
                break

def normaliseList(ll,maxv=10.):
    mmin = np.min(ll)
    llrange = np.max(ll) - mmin
    newll = [(item-mmin)*maxv/llrange for item in ll]
    return newll


def getPairs(start,gap,feature):
    names = ['Increase','Decrease','NoChange']
    scores = []
    arguments = []
    for nn in names:
        _,ss = getFeatureScore(nn,feature)
        scores.extend(ss)
        aa = readArguments('../Convincingness/{}.argument'.format(nn))
        arguments.extend(aa)

    assert len(scores) == len(arguments)
    error = 0.1

    new_scores = normaliseList(np.array(scores))
    print('\n---start: {}+/-{}---\n'.format(start,error))
    arg_pool = []
    for idx,ss in enumerate(new_scores):
        if np.fabs(ss-start) < error \
            and len(word_tokenize(arguments[idx])) >= 10 \
            and '|' not in arguments[idx] \
            and '</' not in arguments[idx]:
            arg_pool.append(arguments[idx])
            #print(arguments[idx])

    for _ in range(10):
        print(random.choice(arg_pool))

    print('\n---end : {}+/-{}---\n'.format(start+gap,error))
    arg_pool = []
    for idx,ss in enumerate(new_scores):
        if np.fabs(start+gap-ss) < error \
                and len(word_tokenize(arguments[idx])) >= 10 \
                and '|' not in arguments[idx] \
                and '</' not in arguments[idx]:
            #print(arguments[idx])
            arg_pool.append(arguments[idx])
    for _ in range(10):
        print(random.choice(arg_pool))




if __name__ == '__main__':
    getTopBottom('Politeness')
    #getPairs(2.5,7.)
