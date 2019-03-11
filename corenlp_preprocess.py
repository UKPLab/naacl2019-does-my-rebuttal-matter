# pip install stanford-corenlp

import os, corenlp
from resources import CORENLP_PATH
os.environ["CORENLP_HOME"] = CORENLP_PATH

class Preprocessor:
    def __init__(self):
        self.client = corenlp.CoreNLPClient(annotators="tokenize ssplit".split())

    def preprocess(self, text):
        ann = self.client.annotate(text)
        return ann

if __name__ == "__main__":
    HOME = "/home/kuznetsov/Projects/ACLC/resources/ACL18/"
    print(Preprocessor().preprocess("A cat in a hat sat on a mat"))
