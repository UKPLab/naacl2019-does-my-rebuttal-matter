import reader
import string
import re
from nltk.tokenize import word_tokenize

PUNCT = tuple(string.punctuation)


def getAllNumbers(ss):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    return rx.findall(ss)


def buildScoreDic(before,after):
    dic = {}

    for ridx,reviews in enumerate([before,after]):
        for rr in reviews:
            if rr._pid not in dic:
                dic[rr._pid] = [{},{}]
            inner_dic = dic[rr._pid][ridx]
            inner_dic[rr._reviewer] = rr._scores['Overall Score']
            inner_dic[rr._reviewer+'_conf'] = rr._scores['Reviewer Confidence']


    return dic

def getReviewIndex(pid,rname,index):
    ll = index[pid]

    names = [i for (i,j) in ll]
    times = [j for (i,j) in ll]

    sorted_time = sorted(times,reverse=False)

    return sorted_time.index(times[names.index(rname)])+1


def getSubTimeDic(reviews):
    dic = {}

    for rr in reviews:
        if rr._pid not in dic:
            dic[rr._pid] = [(rr._reviewer,rr._submit_time)]
        else:
            dic[rr._pid].append((rr._reviewer,rr._submit_time))

    return dic



def getAllReviews(reviews, early_stop=-1):
    ll = []

    for idx, review in reviews.iterrows():
        rr = reader.Review(review)
        ll.append(rr)
        if early_stop>0 and len(ll)>early_stop:
            break

    return ll

def getAllResponses(responses, early_stop=-1):
    ll = []

    for idx, resp in responses.iterrows():
        rr = reader.Response(resp)
        ll.append(rr)
        if early_stop>0 and len(ll)>early_stop:
            break

    return ll

def getReviewsByPid(reviews,pid):
    ll = []
    for rr in reviews:
        if rr._pid == pid:
            ll.append(rr)

    return ll

def getResponseByPid(responses,pid):

    for rr in responses:
        if rr._pid == pid:
            return rr

    return None

def getReviewByPidReviewer(reviews,pid,reviewer):
    for ii,rr in enumerate(reviews):
        if rr._pid == pid and rr._reviewer == reviewer:
            return rr, ii

    return None, None



def remove_stopwords(words, stoplist):
    ''' Remove stop words
    Parameter arguments:
    words = list of words e.g. ['.', 'The', 'boy', 'is', 'playing', '.']

    return:
    list of tokens
    ['boy', 'is', 'playing']
    '''
    return [ token for token in words if not (token.startswith(PUNCT) or token in stoplist)]

def text_normalization(text):
    '''
    Normalize text
    Remove & Replace unnessary characters
    Parameter argument:
    text: a string (e.g. '.... *** New York N.Y is a city...')

    Return:
    text: a string (New York N.Y is a city.)
    '''
    text = re.sub(u'\u201e|\u201c',u'', text)
    text = re.sub(u"\u2022",u'. ', text)
    text = re.sub(u"([.?!]);",u"\\1", text)
    text = re.sub(u'``', u'``', text)
    text = re.sub(u"\.\.+",u" ", text)
    text = re.sub(u"\s+\.",u".", text)
    text = re.sub(u"\?\.",u"?", text)
    text = re.sub(u'[\n\s\t_]+',u' ', text)
    text = re.sub(u"[*]",u"", text)
    text = re.sub(u"\-+",u"-", text)
    text = re.sub(u'^ ',u'', text)
    text = re.sub(u'\u00E2',u'', text)
    text = re.sub(u'\u00E0',u'a', text)
    text = re.sub(u'\u00E9',u'e', text)

    return text


def sent2tokens(sent, language, lower=True):
    '''
    Sentence to stemmed tokens
    Parameter arguments:
    words = list of words e.g. sent = '... The boy is playing.'

    return:
    list of tokens
    ['the', 'boy', 'is', 'playing','.']
    '''
    if lower == True:
        sent = sent.lower()
    sent = text_normalization(sent)
    words = word_tokenize(sent, language)
    return words


def sent2tokens_wostop(sent, stoplist, language):
    '''
    Sentence to tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'

    return:
    list of stemmed tokens without stop words
    ['boys', 'are', 'playing']
    '''

    words = sent2tokens(sent, language)
    tokens = remove_stopwords(words, stoplist)
    return tokens

def sent2stokens_wostop(sent, stemmer, stoplist, language):
    '''
    Sentence to stemmed tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'

    return:
    list of stemmed tokens without stop words
    ['boy', 'are', 'play']
    '''
    tokens = sent2tokens_wostop(sent, stoplist, language)
    return [stemmer.stem(token) for token in tokens]


