import reader
from RebuttalAnalysis.utils import *
import numpy as np
import pandas as pd
from resources import BASE_PATH

'''
Align reviews, their corresponding author responses and corresponding discussions among reviewers.
Sample use: python3 align_review_response.py
'''


if __name__ == '__main__':
    corpus = reader.ACL18Corpus('{}/../consent_data'.format(BASE_PATH))

    after_reviews = getAllReviews(corpus._reviews)
    before_reviews = getAllReviews(corpus._early_reviews)
    responses = getAllResponses(corpus._author_responses)

    dic = buildScoreDic(before_reviews,after_reviews)
    time_index = getSubTimeDic(before_reviews)

    inc = []
    dec = []
    no_change = []
    total_cnt = 0
    # count how many reviewers mention the author responses in discussions

    zero_before_cnt = 0
    zero_before_change_cnt = 0

    for pid in dic:
        before_reviewers = dic[pid][0].keys()
        for br in before_reviewers:
            if '_conf' in br:
                continue
            if br in dic[pid][1]:
                if dic[pid][0][br] == 0:
                    zero_before_cnt += 1
                    if dic[pid][1][br] != 0:
                        zero_before_change_cnt += 1
                    continue
                total_cnt += 1
                ridx = getReviewIndex(pid,br,time_index)
                after_review_text = repr(getReviewByPidReviewer(after_reviews,pid,br)[0]._reviews)
                before_review_text= repr(getReviewByPidReviewer(before_reviews,pid,br)[0]._reviews)
                delta = dic[pid][1][br] - dic[pid][0][br]
                response = getResponseByPid(responses,pid)
                if response is None:
                    resp_text = 'no_permission'
                else:
                    resp_text = response.getResponseToOneReview(ridx)
                discuss = corpus._discussions[corpus._discussions['Reviewer']==br]
                discuss = discuss[discuss['PaperID']==pid]
                if discuss.shape[0] > 0:
                    discuss_text = list(discuss['Text'])[0]
                else:
                    discuss_text = ''
                temp = [pid,br,dic[pid][0][br],dic[pid][1][br]*1.,dic[pid][0]['{}_conf'.format(br)],
                        dic[pid][1]['{}_conf'.format(br)]*1.,before_review_text,after_review_text,resp_text,discuss_text]
                if delta > 0:
                    #inc.append(temp)
                    inc.append(pid)
                elif delta < 0:
                    #dec.append(temp)
                    dec.append(pid)
                else:
                    #no_change.append(temp)
                    no_change.append(pid)

    print('zero before score count {}, among them {} changed'.format(zero_before_cnt,zero_before_change_cnt))

    all_data = None

    df = pd.DataFrame(data=inc,columns=['PID','Reviewer','BeforeScore','AfterScore','BeforeConf','AfterConf','BeforeReview','AfterReview','AuthorResponse','Discussion'])
    df.to_csv('Increase.csv')
    all_data = df

    df = pd.DataFrame(data=dec,columns=['PID','Reviewer','BeforeScore','AfterScore','BeforeConf','AfterConf','BeforeReview','AfterReview','AuthorResponse','Discussion'])
    df.to_csv('Decrease.csv')
    all_data = all_data.append(df,sort=False)

    df = pd.DataFrame(data=no_change,columns=['PID','Reviewer','BeforeScore','AfterScore','BeforeConf','AfterConf','BeforeReview','AfterReview','AuthorResponse','Discussion'])
    df.to_csv('NoChange.csv')
    all_data = all_data.append(df,sort=False)

    all_data.to_csv('all.csv')

    print('{} reviwers changed their reviews, {} increase, {} decrease'.format(
        total_cnt,len(inc),len(dec)
    ))




