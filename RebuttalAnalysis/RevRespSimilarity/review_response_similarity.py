from resources import EMBEDDING_PATH
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from pythonrouge.pythonrouge import Pythonrouge
from RebuttalAnalysis.utils import *


def rouge(hyp, ref, length=1000):
    # 1 - 4, L
    hyp, ref = " ".join(hyp), " ".join(ref)
    ret = Pythonrouge(
        summary_file_exist=False,
        summary=[[hyp]], reference=[[[ref]]],
        n_gram=4, ROUGE_SU4=True, ROUGE_L=True,
        recall_only=True, stemming=True, stopwords=False,
        word_level=True, length_limit=True, length=length,
        use_cf=False, cf=95, scoring_formula='average',
        resampling=True, samples=1000, favor=True, p=0.5).calc_score()

    return ret


def averageEmbedding(model,sentence):
    vecs = []
    for w in sentence: # assume it's already tokenized
        if w in model:
            vecs.append( model[w] )
    if vecs==[]: vecs.append(np.zeros((model.vector_size,)))
    return np.mean(vecs,axis=0)

def cleanReview(review):
    comp = review.split(":")
    stoplist = ['Summary and Contribution','Strengths','Strength argument','Weaknesses','Weakness argument',' (Optional)','Question','Summary','nan','Contribution 1','Contribution 2','Contribution 3']
    content = []
    for cc in comp:
        flag = False
        for ss in stoplist:
            if ss in cc and len(cc)<40:
                flag = True
                break
        if not flag:
            content.append(cc)

    return ' '.join(content)

def getSimilarity(name,stemmer,stoplist,language,model):
    data = pd.read_csv('../../Discussion&Response/{}.csv'.format(name))
    data = data.loc[:,['PID','Reviewer','BeforeReview','AuthorResponse']]
    print('\n\n===old data columns: {}==='.format(data.columns.values))
    print('old data shape {}'.format(data.shape))
    sims = []
    rouge1 = []
    rouge2 = []

    for _,entry in data.iterrows():
        raw_review = repr(entry['BeforeReview']).strip()
        response = repr(entry['AuthorResponse']).strip()
        if response == 'nan' or 'no_permission' in response:
            response = ''

        review = cleanReview(raw_review)
        print('\nreview: {}'.format(review))
        print('response: {}'.format(response))

        if response == '' or review == '':
            sims.append(0.0)
            rouge1.append(0.0)
            rouge2.append(0.0)
        else:
            rgs = rouge(response,raw_review)
            rouge1.append(rgs['ROUGE-1'])
            rouge2.append(rgs['ROUGE-2'])

            review = sent2stokens_wostop(review,stemmer,stoplist,language)
            response = sent2stokens_wostop(response,stemmer,stoplist,language)
            rev_emb = averageEmbedding(model,review)
            resp_emb = averageEmbedding(model,response)
            sim = cosine_similarity(rev_emb.reshape(1,-1),resp_emb.reshape(1,-1))[0][0]
            print(sim,rouge1[-1],rouge2[-1])
            sims.append(sim)

    assert len(sims) == data.shape[0] == len(rouge1) == len(rouge2)
    data['EmbdSimilarity'] = np.array(sims)
    data['Rouge1Similarity'] = np.array(rouge1)
    data['Rouge2Similarity'] = np.array(rouge2)
    return data.loc[:,['PID','Reviewer','EmbdSimilarity','Rouge1Similarity','Rouge2Similarity']]


if __name__ == '__main__':
    stemmer = PorterStemmer()
    language = 'english'
    stoplist = set(stopwords.words(language))
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH)
    names = ['Increase','Decrease','NoChange']
    all_sims = None

    for name in names:
        sims = getSimilarity(name,stemmer,stoplist,language,model)
        sims.to_csv('{}.sim'.format(name))
        if all_sims is None:
            all_sims = sims
        else:
            all_sims = all_sims.append(sims,sort=False)

    all_sims.to_csv('all.sim')
