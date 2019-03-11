import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error

from resources import BASE_PATH

'''
def getReviewFeatures(dic,pid,reviewer,review_list):
    result = []
    other_str = []
    other_wkn = []
    other_qua = []

    for rr in [i for i in dic[pid][0] if '_conf' not in i]:
        if rr == reviewer:
            review,_ = getReviewByPidReviewer(review_list,pid,reviewer)
            str_num = len(review.getStrengths())
            wkn_num = len(review.getWeaknesses())
            qua_num = len(review.getQuestions())
            result.extend([str_num,wkn_num,qua_num])
        else:
            review,_ = getReviewByPidReviewer(review_list,pid,rr)
            str_num = len(review.getStrengths())
            wkn_num = len(review.getWeaknesses())
            qua_num = len(review.getQuestions())
            other_str.append(str_num)
            other_wkn.append(wkn_num)
            other_qua.append(qua_num)

    if len(other_str) > 0:
        result.append(np.mean(other_str))
    else:
        result.append(0.)

    if len(other_wkn) > 0:
        result.append(np.mean(other_wkn))
    else:
        result.append(0.)

    if len(other_qua) > 0:
        result.append(np.mean(other_qua))
    else:
        result.append(0.)

    return result
'''


def getOtherScore(slist):
    return np.max(slist), np.min(slist), np.mean(slist), np.median(slist), np.std(slist)

def getScoreFeatures(scores,pid,reviewer):
    ### get self scores
    entries = scores[(scores['PID']==pid) & (scores['Reviewer']==reviewer)]
    self_score = entries['BeforeScore'].values[0]
    self_conf = entries['BeforeConf'].values[0]
    result = [(self_score,'self_prev'),(self_conf,'self_prev_conf')]

    ### get other scores
    entries = scores[(scores['PID']==pid) & (scores['Reviewer']!=reviewer)]
    other_max, other_min, other_mean, other_median, other_std = getOtherScore(entries['BeforeScore'].values)
    other_conf_max, other_conf_min, other_conf_mean, other_conf_median, other_conf_std = getOtherScore(entries['BeforeConf'].values)

    ### get all scores
    entries = scores[scores['PID']==pid]
    all_max, all_min, all_mean, all_median, all_std = getOtherScore(entries['BeforeScore'].tolist())

    result.extend([(other_mean,"other_mean"),
                   (other_median,"other_median"),
                   (other_min,"other_min"), (other_max,"other_max"), (other_std,"other_dev"),
                   (other_conf_max,"other_conf_max"), (other_conf_min,"other_conf_min"),
                   (other_conf_mean,"other_conf_mean"), (other_conf_median,'other_conf_media'),
                   (other_conf_std,"other_conf_dev"),
                   (other_mean-self_score,'other_mean-self'),
                   (other_median-self_score,'other_median-self'),
                   (other_max-self_score,"other_max-self"),
                   (self_score-other_min,"self-other_min"),
                   (all_mean,"all_mean"),  (all_std,"all_dev"),
                   (all_max,"all_max"),  (all_min,"all_min"), (all_median,'all_median'),
                   (self_score**2,"prev. score**2"),
                   (all_mean-self_score,"all_mean-self"),
                   (all_median-self_score,'all_median-self'),
                   (all_max-self_score,'all_max-self'), (self_score-all_min,'self-all_min')
                    ])

    result,names = [x[0] for x in result],[x[1] for x in result]
    return result, names

def cleanImcompleteEntries(pids,fmatrix,targets):
    assert len(fmatrix) == len(targets) == len(pids)
    remove_idx = []
    for pp in pids:
        if pids.count(pp) < 3:
            idx = [ii for ii in range(len(pids)) if pids[ii]==pp]
            remove_idx.extend(idx)

    newmatrix = []
    newtargets = []
    for ii in range(len(fmatrix)):
        if ii not in remove_idx:
            newmatrix.append(fmatrix[ii])
            newtargets.append(targets[ii])

    return newmatrix, newtargets

def aggregateFeatures(scores,specs,plts,cvcs,sims,onlyBorderLine=True):
    feature_matrix = []
    feature_names = []
    targets = []
    flag = False
    pid_list = []

    for ii,sentry in scores.iterrows():
        entry = []
        pid = sentry['PID']
        reviewer = sentry['Reviewer']

        ### if author response is empty, continue
        if sentry['AuthorResponse'] == 'no_permision' or repr(sentry['AuthorResponse'])=='nan':
            continue
        ### if there's no other peer reviews, continue
        if scores[scores['PID']==pid].shape[0] < 3:
            continue
        ### only borderline cases
        prev_avg = np.mean(scores[scores['PID']==pid]['BeforeScore'].values)
        if onlyBorderLine and (prev_avg < 3 or prev_avg > 4.5):
            continue

        ### response length
        entry.append(np.log(len(word_tokenize(sentry['AuthorResponse']))))

        ### score features
        score_features, score_feature_names = getScoreFeatures(scores,pid,reviewer)
        entry.extend(score_features)

        fnames = ['max','min','mean','median','std']
        ### specificity
        if specs[(specs['PID']==pid) & (specs['Reviewer']==reviewer)].loc[:,fnames].isnull().any().any():
            continue
        spec_features = specs[(specs['PID']==pid) & (specs['Reviewer']==reviewer)].loc[:,fnames].values
        entry.extend(list(spec_features[0]))

        ### politeness
        if plts[(plts['PID']==pid) & (plts['Reviewer']==reviewer)].loc[:,fnames].isnull().any().any():
            continue
        plt_features = plts[(plts['PID']==pid) & (plts['Reviewer']==reviewer)].loc[:,fnames].values
        entry.extend(list(plt_features[0]))

        ### convincingness
        if cvcs[(cvcs['PID']==pid) & (cvcs['Reviewer']==reviewer)].loc[:,fnames].isnull().any().any():
            continue
        cvc_features = cvcs[(cvcs['PID']==pid) & (cvcs['Reviewer']==reviewer)].loc[:,fnames].values
        if np.max(cvc_features) == np.min(cvc_features) == 0:
            continue
        entry.extend(list(cvc_features[0]))

        ### similarity
        if sims[(sims['PID']==pid) & (sims['Reviewer']==reviewer)].loc[:,'EmbdSimilarity'].isnull().any().any():
            continue
        sim_features = sims[(sims['PID']==pid) & (sims['Reviewer']==reviewer)].loc[:,'EmbdSimilarity'].values
        entry.append(sim_features[0])


        if not flag:
            feature_names.append('log_resp_length')
            feature_names.extend(score_feature_names)
            spec_feature_names = ['spec_{}'.format(nn) for nn in fnames]
            feature_names.extend(spec_feature_names)
            plt_feature_names = ['plt_{}'.format(nn) for nn in fnames]
            feature_names.extend(plt_feature_names)
            cvc_feature_names = ['cvc_{}'.format(nn) for nn in fnames]
            feature_names.extend(cvc_feature_names)
            feature_names.append('rev_resp_embd_sim')
            flag = True

        pid_list.append(pid)
        print('pid {}, reviewer {}'.format(pid,reviewer))
        targets.append(sentry['AfterScore'])
        feature_matrix.append(entry)

    feature_matrix, targets = cleanImcompleteEntries(pid_list,feature_matrix,targets)
    print('length: {}'.format(len(targets)))
    print('prev baseline: {}'.format(mean_squared_error(np.array(feature_matrix)[:,1],targets)))
    return np.array(feature_matrix),  np.array(targets), feature_names

if __name__ == '__main__':
    ### score features
    score_data = pd.read_csv('{}/Discussion&Response/all.csv'.format(BASE_PATH))
    score_features = score_data.loc[:,['PID','Reviewer','BeforeScore','AfterScore','BeforeConf','AfterConf','AuthorResponse']]
    ### for specificity scores
    spec_features = pd.read_csv('../Specificity/all_specificity.csv')
    ### for politeness scores
    plt_features = pd.read_csv('../Politeness/all_politeness.csv')
    ### for convincingness
    cvc_features = pd.read_csv('../Convincingness/all_cvc.csv')
    ### for similarity
    sim_features = pd.read_csv('../RevRespSimilarity/all_similarity.csv')

    onlyBorderLine = False

    features, scores, feature_names = aggregateFeatures(score_features,spec_features,plt_features,cvc_features,sim_features,onlyBorderLine)

    dd = pd.DataFrame(columns=feature_names+['score'],data=np.append(features,scores.reshape(-1,1),axis=1))

    if onlyBorderLine:
        dd.to_csv('./borderline_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv')
    else:
        dd.to_csv('./all_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv')

