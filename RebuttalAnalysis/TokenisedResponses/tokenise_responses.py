import pandas as pd
import os
from nltk import sent_tokenize, word_tokenize
from resources import REVIEW_RESPONSE_PATH


def mySentTokenize(texts):
    ori_sent = texts.split('\n')
    cleaned_sent = []
    key_words = {'weakness','question','reply'}

    for ss in ori_sent:
        if ss.strip() == '':
            continue
        elif ss.strip()[-1] == ':' and len(word_tokenize(ss)) <= 10:
            continue
        elif ':' in ss and len(set(word_tokenize(ss.split(':')[0].lower())).intersection(key_words)) > 0 and \
            len(word_tokenize(ss.split(':')[0])) <= 10:
            print(ss)
            cleaned_sent.append(ss.split(':')[1])
        else:
            cleaned_sent.append(ss)

    tokenized_sent = []
    for ss in cleaned_sent:
        tokenized_sent.extend(sent_tokenize(ss))

    return tokenized_sent

def writeSentences(name):
    data = pd.read_csv('{}/{}.csv'.format(REVIEW_RESPONSE_PATH,name))

    output_str = ''
    index_str = ''

    for ii,entry in data.iterrows():
        response = entry['AuthorResponse']
        if pd.isnull(response):
            index_str += '0\n'
        else:
            sents = mySentTokenize(response)
            index_str += '{}\n'.format(len(sents))
            for sen in sents:
                output_str += sen+'\n'

    ff = open(os.path.join('./{}.index'.format(name)),'w')
    ff.write(index_str)
    ff.close()

    ff = open(os.path.join('./{}.sent'.format(name)),'w')
    ff.write(output_str)
    ff.close()


if __name__ == '__main__':
    names = ['Increase','Decrease','NoChange']
    for nn in names:
        print('writing sentences for {}'.format(nn))
        writeSentences(nn)




