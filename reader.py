import os, glob, json, re
import pandas as pd
from corenlp_preprocess import Preprocessor
from dateutil import parser
import math


ASPECTS = ["Overall Score", "Meaningful Comparison", "Originality",
                 "Readability", "Replicability",
                 "Reviewer Confidence",
                 "Soundness/Correctness", "Substance"]
REVIEW_SECTIONS = ["Summary and Contributions","Strengths","Weaknesses","Questions to Authors (Optional)",
                   "Additional Comments (Optional)"]



class Corpus:
    def __init__(self):
        self._papers = None
        self._reviews = None
        self._info = None
        self._preprocessor = None
        self._early_reviews = None
        self._author_responses = None
        self._review_quality = None
        self._discussions = None

    def papers(self):
        return [p[1] for p in sorted(self._papers.items(), key=lambda x: x[0])]

    def get_paper(self, pid):
        return self._papers[pid]

    def get_reviews(self, pid):
        return self._reviews[self._reviews["Submission ID"] == pid]

    def get_early_reviews(self, pid):
        return self._early_reviews[self._early_reviews["Submission ID"] == pid]

    def get_author_responses(self, pid):
        return self._author_responses[self._author_responses["Submission ID"] == pid]

    def get_review_quality(self, pid):
        return self._review_quality[self._review_quality['Submission ID'] == pid]

    def get_info(self, pid):
        return self._info[self._info["Submission ID"] == pid]

    def get_status(self, pid):
        return self._info[self._info["Submission ID"] == pid]["Acceptance Status"].values[0]

    def pids(self):
        return self._papers.keys()

    def preprocess_papers(self):
        def cleanup(x):
            return re.sub("([0-9]+\n)([0-9]+\n)+", " ", x)

        unicodeerr = 0

        if self._preprocessor is None:
            self._preprocessor = Preprocessor()

        pp = self.papers()
        for i in range(len(pp)):
            print("\r{}/{} - err: {}".format(i+1, len(pp), unicodeerr), end="")
            p = pp[i]
            anns = []  # to spare the CoreNLP server, preprocess title, abstract and sections separately
            if p.title is not None:
                anns += [self._preprocessor.preprocess(p.title)]
            if p.abstract is not None:
                anns += [self._preprocessor.preprocess(p.abstract)]
            for s in p.sections:
                try:
                    anns += [self._preprocessor.preprocess("{}\n{}\n".format(s.header, cleanup(s.text)))]
                except UnicodeEncodeError:
                    unicodeerr += 1

            out = []
            for ann in anns:
                for s in ann.sentence:
                    sout = []
                    for t in s.token:
                        sout += [t.word]
                    out += [sout]
            p.ann = out

        print("\nDone, {} unicode errors".format(unicodeerr))

    # TODO preprocess reviews
    # TODO escape formulas, citations etc.
    # TODO advanced preprocessing?..

    @staticmethod
    def _read_papers(papers_src):
        raise NotImplementedError

    @staticmethod
    def _read_reviews(reviews_src):
        raise NotImplementedError

    @staticmethod
    def _read_info(info_src):
        raise NotImplementedError

class ACL18Corpus(Corpus):
    def __init__(self, src):
        super().__init__()
        #self._papers = self._read_papers(os.path.join(src, "parsed"))
        self._reviews = self._read_reviews(os.path.join(src, "After_Rebuttal_Reviews.csv"))
        self._early_reviews = self._read_reviews(os.path.join(src, "Before_Rebuttal_Reviews.csv"))
        #self._info = self._read_info(os.path.join(src, "MetaData", "Submission_Information.csv"))
        self._author_responses = self._read_responses(os.path.join(src, "Author_Response_Information.csv"))
        #self._discussions = self._read_discussions(os.path.join(src,'Message_Board_Information.csv'))
        #self._review_quality = self._read_review_quality(os.path.join(src,'MetaData','Review_Quality_Survey.csv'))

    @staticmethod
    def _read_papers(papers_src):
        papers = {}
        for fn in glob.glob(os.path.join(papers_src, "*.json")):
            if "_Paper" in fn:
                pid = int(os.path.split(fn)[-1].split("_")[0])
                with open(fn) as f:
                    data = json.load(f)
                p = Paper(data, pid)
                papers[pid] = p
        print(f"Read {len(papers)} papers from {os.path.abspath(papers_src)}")
        return papers

    @staticmethod
    def _read_reviews(reviews_src):
        return pd.read_csv(reviews_src)

    @staticmethod
    def _read_responses(responses_src):
        return pd.read_csv(responses_src)

    @staticmethod
    def _read_review_quality(review_quality_src):
        return pd.read_csv(review_quality_src)

    @staticmethod
    def _read_info(info_src):
        return pd.read_csv(info_src)

    @staticmethod
    def _read_discussions(discussions_src):
        return pd.read_csv(discussions_src)



# TODO: not supported so far, update
class PeerReadCorpus(Corpus):
    def __init__(self, src):
        super().__init__()
        self._papers = self._read_papers(os.path.join(src, "parsed_pdfs"))
        self._reviews = self._read_reviews(os.path.join(src, "reviews"))

    @staticmethod
    def _read_papers(papers_src):
        papers = {}
        for fn in glob.glob(os.path.join(papers_src, "*.json")):
            pid = int(os.path.split(fn)[-1].split(".")[0])
            with open(fn) as f:
                data = json.load(f)
            p = Paper(data, pid)
            papers[pid] = p
        print("Read {len(papers)} papers from {os.path.abspath(papers_src)}")
        return papers


    @staticmethod
    def _read_reviews(reviews_src):
        SCORE_COLUMNS = ["APPROPRIATENESS", "CLARITY", "IMPACT",
                         "MEANINGFUL_COMPARISON", "ORIGINALITY", "RECOMMENDATION",
                         "REVIEWER_CONFIDENCE", "SOUNDNESS_CORRECTNESS", "SUBSTANCE"]
        temp = []
        for fn in glob.glob(os.path.join(reviews_src, "*.json")):
            with open(fn) as f:
                data = json.load(f)
                for r in data["reviews"]:
                    r["Submission ID"] = int(data["id"])
                    temp += [r]
        df = pd.DataFrame.from_dict(temp)
        for sc in SCORE_COLUMNS:
            if sc in df.columns:
                df[sc] = pd.to_numeric(df[sc], downcast="integer")
        return df



class Section:
    def __init__(self, header, text):
        self.header = header
        self.text = text

    def __repr__(self):
        return "[{}]".format(self.header)


class Reference:
    def __init__(self, title, author, year, venue, citeregex):
        self.title = title
        self.author = author
        self.year = year
        self.venue = venue
        self.cite = citeregex.replace("\\", "").replace("?", "")

    def __repr__(self):
        return "{} ({}): {}".format(self.author, self.year, self.title)

    def __hash__(self):
        return hash("{} ({}): {}".format(self.author, self.year, self.title))


class Paper:
    def __init__(self, data=None, pid=None):
        if data is None:
            self.pid = None
            self.title = None
            self.abstract = None
            self.sections = None
            self.references = None
            self.ann = None
        else:
            self.pid = pid
            meta = data["metadata"]
            self.title = meta["title"]
            self.abstract = meta["abstractText"]
            self.sections = [Section(s["heading"], s["text"]) for s in meta["sections"] if s["heading"] is not None]
            self.fulltext = "\n".join([s.header+"\n"+s.text for s in self.sections])
            self.ann = None
            self.references = []
            for ref in meta["references"]:
                self.references += [Reference(ref["title"], ref["author"], ref["year"], ref["venue"], ref["citeRegEx"])]
            self.references = sorted(list(set(self.references)), key=lambda x: x.cite)

    def __str__(self):
        return "{}: {}".format(self.pid, self.title)

    def __repr__(self):
        return "{}: {}".format(self.pid, self.title)


class Review:
    def __init__(self, review):

        self._pid = review['Submission ID']
        self._reviewer = review['Reviewer Username']
        self._submit_time = parser.parse(review['First Submission Time'])
        self._scores = {}
        self._reviews = {}

        for aspect in ASPECTS:
            self._scores[aspect] = review[aspect]
            if math.isnan(self._scores[aspect]):
                self._scores[aspect] = 0.
        for section in REVIEW_SECTIONS:
            self._reviews[section] = review[section]

        #TODO: add parse contributions, questions, etc., so as to remove self._reviews and add different sections

    def getStrengths(self):
        return Review.parseArgumentsQuestions(self._reviews['Strengths'])

    def getWeaknesses(self):
        return Review.parseArgumentsQuestions(self._reviews['Weaknesses'])

    def getQuestions(self):
        return Review.parseArgumentsQuestions(self._reviews['Questions to Authors (Optional)'])

    @staticmethod
    def parseArgumentsQuestions(text):
        if pd.isnull(text):
            return []

        arg_list = []
        temp_arg = ''
        flag = False

        for line in text.split('\n'):

            if 'Strength argument' in line:
                flag = True
                if temp_arg != '':
                    arg_list.append(temp_arg)
                temp_arg = ''.join(line.split(':')[1:]).strip()
            elif 'Weakness argument' in line:
                flag = True
                if temp_arg != '':
                    arg_list.append(temp_arg)
                temp_arg = ''.join(line.split(':')[1:]).strip()
            elif len(re.compile('question\s?\d+').findall(line.lower())) > 0:
                flag = True
                if temp_arg != '':
                    arg_list.append(temp_arg)
                temp_arg = ''.join(line.split(':')[1:]).strip()
            elif flag and line.strip() != '':
                temp_arg += line + '\n'

        if temp_arg.strip() != '':
            arg_list.append(temp_arg)

        #if len(arg_list) == 0:
            #return [text]

        return arg_list



class Response:

    def __init__(self,response):
        self._pid = response['Submission ID']

        if not pd.isna(response['Responses To Individual Reviews']):
            self._response_to_reviews = response['Responses To Individual Reviews']
        else:
            self._response_to_reviews = ''

        if not pd.isna(response['General Response to Reviews']):
            self._general_response = response['General Response to Reviews']
        else:
            self._general_response = ''

        if not pd.isna(response['Response to Chairs']):
            self._response_to_chairs = response['Response to Chairs']
        else:
            self._response_to_chairs = ''

    def getResponseToOneReview(self,idx):
        lines = self._response_to_reviews.split('\n')
        str = ''
        flag = False

        for line in lines:
            if 'RESPONSE TO REVIEW #{}'.format(idx) in line:
                flag = True
            elif flag:
                if 'RESPONSE TO REVIEW #{}'.format(idx+1) in line:
                    break
                else:
                    str += line+'\n'

        return str






