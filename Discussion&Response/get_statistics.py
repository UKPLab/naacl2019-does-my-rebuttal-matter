import reader
from RebuttalAnalysis.utils import *
from resources import BASE_PATH
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def getArgNumStatistics(corpus):
    after_reviews = getAllReviews(corpus._reviews)
    stg_nums = [len(ee.getStrengths()) for ee in after_reviews]# if len(ee.getStrengths())>0]
    wek_nums = [len(ee.getWeaknesses()) for ee in after_reviews]# if len(ee.getWeaknesses())>0]
    que_nums = [len(ee.getQuestions()) for ee in after_reviews]# if len(ee.getQuestions())>0]

    print('strength num {}, max {}, min {}, mean {}, median {}, stdev {}'.format(len(stg_nums), np.max(stg_nums), np.min(stg_nums), np.mean(stg_nums), np.median(stg_nums), np.std(stg_nums)))
    print('weakness num {}, max {}, min {}, mean {}, median {}, stdev {}'.format(len(wek_nums), np.max(wek_nums), np.min(wek_nums), np.mean(wek_nums), np.median(wek_nums), np.std(wek_nums)))
    print('question num {}, max {}, min {}, mean {}, median {}, stdev {}'.format(len(que_nums), np.max(que_nums), np.min(que_nums), np.mean(que_nums), np.median(que_nums), np.std(que_nums)))


def getArgLengStatistics(corpus):
    after_reviews = getAllReviews(corpus._reviews)
    ss = []
    ws = []
    qs = []
    for entry in after_reviews:
        ss.extend(entry.getStrengths())
        ws.extend(entry.getWeaknesses())
        qs.extend(entry.getQuestions())

    ss_leng = [len(word_tokenize(s)) for s in ss]
    ws_leng = [len(word_tokenize(s)) for s in ws]
    qs_leng = [len(word_tokenize(s)) for s in qs]

    print('strength arg leng: max {}, min {}, mean {}, median {}, std {}'.format(np.max(ss_leng), np.min(ss_leng), np.mean(ss_leng), np.median(ss_leng), np.std(ss_leng)))
    print('weakness arg leng: max {}, min {}, mean {}, median {}, std {}'.format(np.max(ws_leng), np.min(ws_leng), np.mean(ws_leng), np.median(ws_leng), np.std(ws_leng)))
    print('question arg leng: max {}, min {}, mean {}, median {}, std {}'.format(np.max(qs_leng), np.min(qs_leng), np.mean(qs_leng), np.median(qs_leng), np.std(qs_leng)))

def getRebuttalStatistics(corpus):
    resp = getAllResponses(corpus._author_responses)
    rr_list = []
    for rr in resp:
        rrs = [rr.getResponseToOneReview(ii) for ii in [1,2,3,4,5]]
        rrcs = [rr.replace('Reply to weakness argument','') for rr in rrs]
        rr_list.append(' '.join(rrcs))
    leng = [len(word_tokenize(repr(rr))) for rr in rr_list if len(word_tokenize(repr(rr)))<=1000 ]
    print('resp length: max {}, min {}, mean {}, median {}, stdev {}'.format(np.max(leng), np.min(leng), np.mean(leng), np.median(leng), np.std(leng)))

def getScoreChangeStatistics():
    inc = pd.read_csv('Increase.csv')
    dec = pd.read_csv('Decrease.csv')
    noc = pd.read_csv('NoChange.csv')
    all = pd.read_csv('all.csv')


    all_text = all['AuthorResponse'].tolist()
    pids = all['PID'].tolist()
    assert len(all_text) == len(pids)
    goodpid = []
    for ii in range(len(pids)):
        if 'no_permission' not in repr(all_text[ii]) :
            goodpid.append(pids[ii])
    print('{} papers opted in their author responses '.format(len(set(goodpid))))

    print('inc size ', inc.shape[0])
    print('dec size ', dec.shape[0])
    print('noc size ', noc.shape[0])
    print('all size ', all.shape[0])

    print('inc before avg score {}, after avg score {}'.format(
        np.mean(inc['BeforeScore'].tolist()),np.mean(inc['AfterScore'].tolist()),
    ))
    print('dec before avg score {}, after avg score {}'.format(
        np.mean(dec['BeforeScore'].tolist()),np.mean(dec['AfterScore'].tolist()),
    ))
    print('noc score {}'.format(
        np.mean(noc['BeforeScore'].tolist()),
    ))
    print('all before avg score {}, after avg score {}'.format(
        np.mean(all['BeforeScore'].tolist()),np.mean(all['AfterScore'].tolist()),
    ))

    ipids = set(inc['PID'].tolist())
    dpids = set(dec['PID'].tolist())
    print('both inc and dec: ', len(ipids.intersection(dpids)))
    left_pids = set(all['PID'].tolist())-ipids.union(dpids)
    all_pids = set(all['PID'].tolist())

    acc_data = pd.read_csv('Submission_Information.csv')
    iacc = [acc_data[acc_data['PID']==pp]['AcceptanceStatus'].values[0] for pp in ipids]
    dacc = [acc_data[acc_data['PID']==pp]['AcceptanceStatus'].values[0] for pp in dpids]
    nacc = [acc_data[acc_data['PID']==pp]['AcceptanceStatus'].values[0] for pp in left_pids]
    aacc = [acc_data[acc_data['PID']==pp]['AcceptanceStatus'].values[0] for pp in all['PID'].tolist()]
    bacc = [acc_data[acc_data['PID']==pp]['AcceptanceStatus'].values[0] for pp in ipids.intersection(dpids)]

    print('inc acc num {}, total pid {}'.format(len([ss for ss in iacc if 'Accept' in ss]), len(ipids)))
    print('dec acc num {}, total pid {}'.format(len([ss for ss in dacc if 'Accept' in ss]), len(dpids)))
    print('both acc num {}, total pid {}'.format(len([ss for ss in bacc if 'Accept' in ss]), len(ipids.intersection(dpids))))
    print('noc acc num {}, total pid {}'.format(len([ss for ss in nacc if 'Accept' in ss]), len(left_pids)))
    print('all acc num {}, total pid {}'.format(len([ss for ss in aacc if 'Accept' in ss]), len(all_pids)))

    inc_text = set(inc['AuthorResponse'].tolist())
    dec_text = set(dec['AuthorResponse'].tolist())
    noc_text = set(noc['AuthorResponse'].tolist())
    all_text = set(all['AuthorResponse'].tolist())

    itl = [len(word_tokenize(tt.replace('Reply to weakness argument',''))) for tt in inc_text if 'no_permission' not in tt]
    dtl = [len(word_tokenize(tt.replace('Reply to weakness argument',''))) for tt in dec_text if 'no_permission' not in tt]
    ntl = [len(word_tokenize(repr(tt).replace('Reply to weakness argument',''))) for tt in noc_text if 'no_permission' not in repr(tt)]
    atl = [len(word_tokenize(repr(tt).replace('Reply to weakness argument',''))) for tt in all_text if 'no_permission' not in repr(tt)]

    print('resp num: inc {}, dec {}, noc {}, all {}'.format(
        len(itl), len(dtl), len(ntl), len(atl)
    ))

    print('inc token num {} std {}, dec token num {} std {}, \n'
          'nochange token num {} std {}, all token num {} std {}'.format(
        np.mean(itl),np.std(itl), np.mean(dtl),np.std(dtl),
        np.mean(ntl), np.std(ntl), np.mean(atl), np.std(atl)
    ))


def getBaselinePrediction():
    data = pd.read_csv('all.csv')
    bscores = []
    ascores = []

    bbor_scores = []
    abor_scores = []
    for ii,entry in data.iterrows():
        bsl = entry['BeforeScore']
        asl = entry['AfterScore']
        bscores.append(bsl)
        ascores.append(asl)

        if bsl > 3 and bsl < 4.5:
            bbor_scores.append(bsl)
            abor_scores.append(asl)

    print('all length: {}'.format(len(bscores)))
    print('all', mean_squared_error(bscores,ascores))
    print('border length: {}'.format(len(bbor_scores)))
    print('border', mean_squared_error(bbor_scores,abor_scores))

def getAccScoreDistribution():
    all_reviews = pd.read_csv('all.csv')
    acc_info = pd.read_csv('Submission_Information.csv')

    pids = set(all_reviews['PID'].tolist())
    ra = [acc_info[acc_info['PID']==pp]['AcceptanceStatus'].values[0] for pp in pids]
    accrej = [aa.split('-')[0].strip() for aa in ra]
    scores = [np.mean(all_reviews[all_reviews['PID']==pp]['AfterScore']) for pp in pids]

    acc = [scores[ii] for ii,status in enumerate(accrej) if status=='Accept']
    rej = [scores[ii] for ii,status in enumerate(accrej) if status=='Reject']
    acc_hist = np.histogram(acc,bins=np.linspace(1,6,11))
    rej_hist = np.histogram(rej,bins=np.linspace(1,6,11))
    print(acc_hist[0], acc_hist[1])

    p1 = plt.bar(rej_hist[1][:-1], rej_hist[0], 0.25)
    p2 = plt.bar(acc_hist[1][:-1], acc_hist[0], 0.25, bottom=rej_hist[0])
    font_size = 18
    plt.rc('font',size=font_size)
    plt.ylabel('#Submission',fontsize=font_size)
    plt.xlabel('After-rebuttal average OVAL',fontsize=font_size)
    plt.legend((p1[0], p2[0]), ('Reject', 'Accept'))
    plt.show()

if __name__ == '__main__':
    corpus = reader.ACL18Corpus('{}/../consent_data'.format(BASE_PATH))
    #getArgNumStatistics(corpus)
    #getArgLengStatistics(corpus)
    getRebuttalStatistics(corpus)
    #getScoreChangeStatistics()
    #getBaselinePrediction()
    #getAccScoreDistribution()

