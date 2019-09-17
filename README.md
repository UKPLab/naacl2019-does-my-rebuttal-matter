# Does my Rebuttal Matter? Insights From a Major NLP Conference 

This repository contains selected code and data for our NAACL 2019 long paper on [Does my Rebuttal Matter?](https://arxiv.org/pdf/1903.11367.pdf).

## Citation

```
@inproceedings{Gao:2019:NAACL,
            title = {Does my Rebuttal Matter? Insights From a Major NLP conference},
            year = {2019},
          author = {Yang Gao and Steffen Eger and Ilia Kuznetsov and Iryna Gurevych and Yusuke Miyao},
       booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
           month = {Februar},
         journal = {NAACL 2019},
             url = {http://tubiblio.ulb.tu-darmstadt.de/111643/}
}
```
> **Abstract:** Peer review is a core element of the scientific
process, particularly in conference-centered
fields such as ML and NLP. However, only
few studies have evaluated its properties empirically. Aiming to fill this gap, we present
a corpus that contains over 4k reviews and
1.2k author responses from ACL-2018. We
quantitatively and qualitatively assess the corpus. This includes a pilot study on paper
weaknesses given by reviewers and on quality of author responses. We then focus on
the role of the rebuttal phase, and propose
a novel task to predict after-rebuttal (i.e., final) scores from initial reviews and author responses. Although author responses do have
a marginal (and statistically significant) influence on the final scores, especially for borderline papers, our results suggest that a reviewer’s final score is largely determined by
her initial score and the distance to the other
reviewers’ initial scores. In this context, we
discuss the conformity bias inherent to peer
reviewing, a bias that has largely been overlooked in previous research. We hope our
analyses will help better assess the usefulness
of the rebuttal phase in NLP conferences  

Contact person: Yang Gao gao@ukp.informatik.tu-darmstadt.de, Steffen Eger, Ilia Kuznetsov

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 

## Project Description
This project includes three major parts: 
* the ACL-2018 Review dataset, 
* our quanlitative analyses on the dataset, and 
* our code for predicting whether the reviewer will increase/decrease/remain her overall score after rebuttal. 

According to the data sharing terms and conditions of ACL-2018, the opted-in reviews will be publically available no earlier than two years after the conference took place. We will publish the dataset in this project once it is allowed.

## Requirement
* Python 3 (tested with Python 3.6 on Ubuntu 16.04)
* libraries in requirement.txt: pip install -r requirement.txt (the use of virtual environment is recommended)

## Project Structure
* Discussion&Response: includes the csv files of the opted-in reviews, submissions information (e.g. paper ids and acceptance/rejection decisions) and author responses. Will be published once allowed.
* RebuttalAnalysis: code used to build the after-rebuttal score predictor. 'classify_after_label.py' includes the main function for the predictor. 'predict_after_score.py' builds a regression to predict the after-rebuttal score for each reviewer (its results are not reported in the orginal paper).

## Word embeddings
We provide word2vec word embeddings trained on Machine Learning (cs.LG) and NLP (cs.CL) Arxiv papers. The embeddings are available from 
https://public.ukp.informatik.tu-darmstadt.de/naacl2019-does-my-rebuttal-matter

## Log-likelihood ratio (LLR) for two datasets

To run a LLR test to determine the most "unusual" words in one corpus relative to another, run
```
python3 getMostFrequent.py accepted.txt rejected.txt 100 2

```
This returns the 100 most unusual bigrams of `accepted.txt` relative to `rejected.txt`. The two input files are plain text files with words separated by white space.

## NOTE/ERRATUM: In Fig. 2 in the [paper](https://arxiv.org/pdf/1903.11367.pdf), "RPB" stands for "Reproducibility", an aspect score in the review template. [Yuval Pinter](www.yuvalpinter.com) kindly informed us that we did not explain the meaning of "RPB" in the paper. We apologise for this ignorance.


