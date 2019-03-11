# SAMPLE USAGE:
# python3 step2_featureSelction_regression.py
# before running, install all packages in requirement.txt

import sys

sys.path.append('../..')

import operator
import argparse
import numpy as np
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import random


def selectByPCA(features,fnames,target_num=10):
    pca = PCA(target_num,copy=True)
    fit = pca.fit_transform(features)
    return fit, None


def selectByVIF(features, fnames, target_num=10, threshold=5.):
    variables = np.array([i for i in range(features.shape[1])])
    cnt = 0
    for i in np.arange(0, len(variables)):
        cnt += 1
        vif = [variance_inflation_factor(features[:,variables], ix) for ix in range(len(variables))]
        #print('round {}: {}'.format(cnt,vif))
        maxloc = vif.index(max(vif))
        if len(variables) <= target_num or max(vif) < threshold:
            break
        else:
            variables = np.delete(variables,maxloc)

    #print('Remaining variables ({}):'.format(len(variables)))
    #for nn in variables:
        #print(fnames[nn])
    return features[:,variables], fnames[variables]


def addRebuttalScoreFeature(features,names):
    self_prevs = features[:,[nn=='all_max' for nn in names]]
    added = [max(ss[0],4.) for ss in self_prevs]
    return np.append(np.array(added).reshape(-1,1),features,axis=1), np.append('rebuttal_score',names)


def randomDownSample(labels, target_lable='nc', left_ratio=0.3):
    idx = []
    for ii in range(len(labels)):
        if labels[ii] != target_lable:
            idx.append(True)
        else:
            idx.append(random.random()<=left_ratio)

    return np.array(idx)


def myCrossValidation(clf,features,labels,cv=50,dev_ratio=0.2):
    window_size = int(len(labels)/cv)
    pointer = 0
    class_names = ['inc','dec','nc','macroAvg']
    metric_names = ['pre','rec','f1']
    results = {}
    for cn in class_names:
        for mn in metric_names:
            results['{}-{}'.format(cn,mn)] = []

    for fold in range(cv):
        avai_idx = np.array([not(ii>=pointer and ii<pointer+window_size) for ii in range(len(labels))])
        test_idx = np.array([ii>=pointer and ii<pointer+window_size for ii in range(len(labels))])
        avai_features = features[avai_idx]
        avai_labels = labels[avai_idx]

        #train_features = avai_features[range(0,int(len(avai_labels)*(1-dev_ratio)))]
        #train_labels = avai_labels[range(0,int(len(avai_labels)*(1-dev_ratio)))]
        #dev_features = avai_features[range(int(len(avai_labels)*(1-dev_ratio)), len(avai_labels))]
        #dev_labels = avai_labels[range(0,int(len(avai_labels)*(1-dev_ratio)), len(avai_labels))]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        ### Down Sampling
        lnames,lcounts = np.unique(avai_labels,return_counts=True)
        lnames = list(lnames)
        lcounts = list(lcounts)
        nc_num = lcounts[lnames.index('nc')]
        inc_dec_num = lcounts[lnames.index('inc')] + lcounts[lnames.index('dec')]
        sampled_idx = randomDownSample(avai_labels,left_ratio=inc_dec_num*1./nc_num)
        clf.fit(avai_features[sampled_idx],avai_labels[sampled_idx])

        pre, rec, f1, _ = precision_recall_fscore_support(test_labels,clf.predict(test_features),labels=class_names[:-1])
        for mi,metric in enumerate([pre,rec,f1]):
            for ni,nn in enumerate(metric):
                results['{}-{}'.format(class_names[ni],metric_names[mi])].append(nn)
        results['macroAvg-pre'].append(np.mean(pre))
        results['macroAvg-rec'].append(np.mean(rec))
        results['macroAvg-f1'].append(np.mean(f1))

        pointer += window_size

    return results


def getLabels(data_path):
    data = pd.read_csv(data_path,usecols=['self_prev','score'])
    labels = []
    prev_scores = np.array(data['self_prev'].tolist())
    scores = np.array(data['score'].tolist())
    delta = scores-prev_scores
    for ii in range(len(scores)):
        if delta[ii] > 0:
            cn = 'inc'
        elif delta[ii] < 0:
            cn = 'dec'
        else:
            cn = 'nc'
        labels.append(cn)
    return np.array(labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_num',action="store",dest="fn",nargs='?',default=-1)
    parser.add_argument("--classifier",action="store",dest="clf",default="log-reg",nargs='?',help="gp|log-reg|forest|tree|svc_lin|svc_poly|svc_rbf|svc_sigmoid")
    parser.add_argument("--feature-selector",action="store",dest="fs",default="none",nargs='?',help='pca or vif')
    parser.add_argument("--feature-set",action="store",dest="fst",
                        default="length-similarity-opinion-politeness-specificity-convincingness",nargs='?')
                        #default="opinion",nargs='?')
    parser.add_argument("--data",action="store",dest="data",
                        default="./AggregateFeatures/borderline_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv",nargs='?')
    #parser.set_defaults(conll=False)

    parsed_args = parser.parse_args(sys.argv[1:])

    print(":::",parsed_args.fn,parsed_args.clf,parsed_args.fs,parsed_args.fst)

    cv = 10
    feature_num = int(parsed_args.fn)
    clf_type = parsed_args.clf
    feature_selector = parsed_args.fs
    feature_set_name = parsed_args.fst
    data_path = parsed_args.data

    ### read features and scores
    length_features = [1]
    opinion_features = list(range(2,28))
    specificity_features = list(range(28,33))
    politeness_features = list(range(33,38))
    convincingness_features = list(range(38,43))
    similarity_features = [43]
    out_feature = [44]

    h = {"opinion":opinion_features,"politeness":politeness_features,"specificity":specificity_features,"length":length_features,'convincingness':convincingness_features,'similarity':similarity_features,'length':length_features}

    feature_set = []
    for x in feature_set_name.split("-"):
          feature_set += h[x] 
      
    if feature_num==-1:
      feature_num = len(feature_set)
    print(len(feature_set))

    ### get classification labels
    labels = getLabels(data_path)
    print('\n===Label Distribution===')
    lnames, cnts = np.unique(labels,return_counts=True)
    for idx in range(len(lnames)):
        print('{} : {} ({}%)'.format(lnames[idx],cnts[idx],cnts[idx]*100./sum(cnts)))

    data = pd.read_csv(data_path,usecols=feature_set+out_feature)
    print("-->",data_path,data.columns.values)

    ### get target labels
    class_names = ['inc','dec','nc']
    feature_names= np.array(list(data.columns.values)[:-1])
    print(feature_names)
    matrix = np.array(data.values)
    features = matrix[:,:-1]

    ### when using score alone, use other_mean-self and prev_self
    ### when not using score features, activate long_resp_length
    ### simple model
    #wanted_features = ['other_mean-self','self-other_min',
                       #'rev_resp_embd_sim',
                       #'plt_max', #'plt_median','plt_min',
                       #'cvc_max',#'cvc_min','cvc_mean',
                       #'spec_median','spec_max',#'spec_min',
                       #'log_resp_length'
                       #] #all case
    wanted_features = ['other_mean-self','self-other_min',
                       'rev_resp_embd_sim',
                       'plt_max', 'plt_median','plt_max',
                       'cvc_min', 'cvc_max',#'cvc_mean',
                       'spec_median',#'spec_max','spec_min',
                       #'log_resp_length'
                       ] #borderline case
    features = features[:,np.array([feature_names[ii] in wanted_features for ii in range(len(feature_names))])]
    feature_names = np.array(wanted_features)
    ### simple model
    features = StandardScaler().fit_transform(features)

    if feature_selector == 'vif' and features.shape[1] > 1:
        features, feature_names = selectByVIF(features, feature_names, feature_num)
    elif feature_selector == 'pca' and features.shape[1] > 1:
        features, feature_names = selectByPCA(features,feature_names,feature_num)

    '''
    ### majority baseline:
    maj_labels = ['nc']*len(labels)
    pre, rec, f1, _ = precision_recall_fscore_support(labels,maj_labels,labels=class_names)
    print('\n===MAJORITY BAELINE===')
    for ii,cn in enumerate(class_names):
        print('---CLASS {}'.format(cn.upper()))
        print('precision {}, recall {}, F1 {}'.format(pre[ii], rec[ii], f1[ii]))
    print('---Macro Avg---')
    print('precision {}, recall {}, F1 {}'.format(np.mean(pre), np.mean(rec), np.mean(f1)))

    ## random baseline:
    f1_list = [[],[],[]]
    p_list = [[],[],[]]
    r_list = [[],[],[]]
    for _ in range(500):
        rnd_labels = [random.choice(['nc','inc','dec']) for ii in range(len(labels))]
        pre, rec, f1, _ = precision_recall_fscore_support(labels,rnd_labels,labels=class_names)
        for ii,cn in enumerate(class_names):
            p_list[ii].append(pre[ii])
            r_list[ii].append(rec[ii])
            f1_list[ii].append(f1[ii])
    print('\n===Random BAELINE===')
    for ii,cn in enumerate(class_names):
        print('---CLASS {}'.format(cn.upper()))
        print('precision {}, recall {}, F1 {}'.format(np.mean(p_list[ii]), np.mean(r_list[ii]), np.mean(f1_list[ii])))
    print('---Macro Avg---')
    print('precision {}, recall {}, F1 {}'.format(np.mean(p_list[0]+p_list[1]+p_list[2]), np.mean(r_list[0]+r_list[1]+r_list[2]), np.mean(f1_list[0]+f1_list[1]+f1_list[2])))
    exit(111)
    '''

    ### cross-validation
    if clf_type == 'gp':
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #clf = GaussianProcessClassifier(kernel=kernel)
        clf = GaussianProcessClassifier()
    elif clf_type == 'svc_rbf':
        clf = SVC(kernel='rbf')
    elif clf_type == 'svc_lin':
        clf = SVC(kernel='linear')
    elif clf_type == 'svc_poly':
        clf = SVC(kernel='poly')
    elif clf_type == 'svc_sigmoid':
        clf = SVC(kernel='sigmoid')
    elif clf_type == 'forest':
        clf = RandomForestClassifier()
    elif clf_type == 'tree':
        clf = DecisionTreeClassifier()
    else:
        clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
        #clf = sklearn.linear_model.LogisticRegression()

    all_results = None
    repeat = 100
    feat_imp_dic = {}
    feat_weights_dic = {}
    for ii in range(repeat):
        if (ii+1)%500 == 0:
            print(ii+1)
        ### shuffle the order of the features and training examples
        indices = np.random.permutation(features.shape[0])
        features = features[indices]
        labels = labels[indices]

        ### shuffling the features
        indices = np.random.permutation(features.shape[1])
        if feature_selector != 'pca':
            feature_names = feature_names[indices]
        features = features[:,indices]

        cv_results = myCrossValidation(clf,features,labels,cv)
        if clf_type == 'forest':
            ww = clf.feature_importances_
            for ii in range(len(feature_names)):
                feat_imp_dic[feature_names[ii]] = feat_imp_dic.get(feature_names[ii],0) + ww[ii]
        elif clf_type == 'log-reg' and feature_selector == 'none':
            classes = list(clf.classes_)
            for i,cc in enumerate(classes):
                if cc not in feat_weights_dic:
                    feat_weights_dic[cc] = {}
                for j, ff in enumerate(feature_names):
                    feat_weights_dic[cc][ff] = feat_weights_dic[cc].get(ff,0) + clf.coef_[i][j]

        if all_results is None:
            all_results = cv_results.copy()
        else:
            for metric in cv_results:
                all_results[metric].extend(cv_results[metric])

    if clf_type == 'forest':
        print('\n---Features Importance---')
        sorted_dic = sorted(feat_imp_dic.items(), key=operator.itemgetter(1), reverse=True)
        for entry in sorted_dic:
            print('{}\t\t{}'.format(entry[0],entry[1]))
    elif clf_type == 'log-reg' and feature_selector == 'none':
        print('\n---Features Weights---')
        for cc in feat_weights_dic:
            print('\n--{}--'.format(cc))
            for ff in feat_weights_dic[cc]:
                print('{}: {}'.format(ff,feat_weights_dic[cc][ff]*1./repeat))

    print('\n===Repeat {} times {} {}-Fold Cross Validation==='.format(repeat,clf_type,cv))
    for metric in cv_results:
        print('{} : mean {}, std {}'.format(metric,np.mean(cv_results[metric]),np.std(cv_results[metric])))




