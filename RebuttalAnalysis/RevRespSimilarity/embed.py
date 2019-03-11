import numpy as np
from gensim.models import KeyedVectors
from resources import EMBEDDING_PATH
from sklearn.metrics.pairwise import cosine_similarity

def averageEmbedding(model,sentence,lowercase=True):
    if lowercase: sentence = sentence.lower()
    vecs = []
    for w in sentence.split(): # assume it's already tokenized
        if w in model:
            vecs.append( model[w] )
    if vecs==[]: vecs.append(np.zeros((model.vector_size,)))
    return np.mean(vecs,axis=0)


if __name__ == "__main__":

    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH)
    avg_list = []
    for sentence in ['this is a nice apple .','this apple is quite good.']:
        avg = averageEmbedding(model,sentence)
        avg_list.append(avg.reshape(1,-1))
        print(" ".join([str(x) for x in list(avg)]))

    print('similarity: {}'.format(cosine_similarity(avg_list[0],avg_list[1])))

