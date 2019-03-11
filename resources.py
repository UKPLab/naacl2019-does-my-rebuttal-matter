import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
REVIEW_RESPONSE_PATH = os.path.join(BASE_PATH,'Discussion&Response')
EMBEDDING_PATH = os.path.join('{}/../embeddings/arxiv/12312017/arxiv_cs.CL_size300_window5_skip.txt'.format(BASE_PATH))
CORENLP_PATH = "/home/gao/Library/NLP-Related/Stanford/stanford-corenlp-full-2015-12-09"