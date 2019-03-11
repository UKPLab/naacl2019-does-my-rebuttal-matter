import sys,numpy as np
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import pandas as pd
from RebuttalAnalysis.utils import text_normalization

# Based on this:
# http://ucrel.lancs.ac.uk/llwizard.html
# See Dunning 1993, Accurate Methods for the Statistics of Surprise and Coincidence

def readText(texts,ngram=1,lower=True):
    tokenizer = RegexpTokenizer(r'\w+')
    docFreq={}
    termFreq={}
    curWords=set()
    ndocs=0
    for doc in texts:
        doc = text_normalization(doc)
        for line in sent_tokenize(doc):
            line = line.strip()
            if lower: line=line.lower()
            text = ["SOS"]*(ngram-1)+tokenizer.tokenize(line)+["EOS"]*(ngram-1)
            for idx in range(len(text)):
                if text[idx].isdigit():
                    text[idx] = 'DIGIT'
            for iw in range(0,len(text)-ngram+1):
                gram = tuple(text[iw:iw+ngram])
                termFreq[gram] = termFreq.get(gram,0)+1
                curWords.add(gram)

        for w in curWords: docFreq[w] = docFreq.get(w,0)+1
        ndocs+=1
        curWords=set()

    return termFreq,docFreq,ndocs


def getFrequentNgrams(inc_text,dec_text,ng,top_num):
    accepted,acc_doc,acc_ndoc = readText(inc_text,ngram=ng)
    rejected,rej_doc,rej_ndoc = readText(dec_text,ngram=ng)

    n_accepted = sum(accepted.values())
    n_rejected = sum(rejected.values())
    n = n_accepted+n_rejected

    llr={}
    k=0

    for w in accepted:
        aw = accepted.get(w,0.5)
        rw = rejected.get(w,0.5)

        idf_a = np.log(acc_ndoc*1.0/acc_doc.get(w,0.5))
        idf_r = np.log(rej_ndoc*1.0/rej_doc.get(w,0.5))

        if acc_doc[w]<7: continue

        #print(idf_a,idf_r,acc_ndoc,rej_ndoc)

        #aw = aw/idf_a
        #rw = rw/idf_r
  
        ea = n_accepted*(aw+rw)/n
        er = n_rejected*(aw+rw)/n

        #print(aw,rw,n_accepted,n_rejected);
 
        my_llr = 2*(aw*np.log(aw/ea)+rw*np.log(rw/er))
        if aw/n_accepted>rw/n_rejected: sign=1
        else: sign=-1
        llr[w] = (sign*my_llr,aw,rw,n_accepted,n_rejected)

        #print(llr[w])
        k+=1
        #if k>1: break

    s = [(k, llr[k]) for k in sorted(llr, key=llr.get, reverse=True)]
    for k,v in s[:top_num]:
        kt = " ".join(k)
        print(kt,v)

if __name__ == '__main__':
    inc_data = pd.read_csv('Increase.csv')
    inc_text = [tt for tt in list(inc_data['AuthorResponse'].values) if repr(tt) != 'nan']

    dec_data = pd.read_csv('Decrease.csv')
    dec_text = [tt for tt in list(dec_data['AuthorResponse'].values) if repr(tt) != 'nan']

    ng = 3
    top_num = 20

    print('\n=====Top {} {}-gram in good responses'.format(top_num,ng))
    getFrequentNgrams(inc_text,dec_text,ng,top_num)

    print('\n\n=====Top {} {}-gram in bad responses'.format(top_num,ng))
    getFrequentNgrams(dec_text,inc_text,ng,top_num)




