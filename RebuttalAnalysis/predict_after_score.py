# SAMPLE USAGE:
# python3 step2_featureSelction_regression.py
# before running, install all packages in requirement.txt

import sys

sys.path.append('../..')

import argparse
import numpy as np
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import math



def printCorrelation(scores,features,feature_names):
    print('\n\n===== Pearson Correlation Between Features and Scores =====')
    for col_id in range(len(feature_names)):
        (rho, pvalue) = pearsonr(scores,features[:,col_id])
        print('{}: {}'.format(feature_names[col_id],rho))

    print('\n\n===== Pearson Correlation Between Features=====')
    for ii in range(len(feature_names)-1):
        for jj in range(ii+1,len(feature_names)):
            (rho, pvalue) = pearsonr(features[:,jj],features[:,ii])
            print('{} and {}: {}'.format(feature_names[ii],feature_names[jj],rho))

    print(' ')


def selectByREF(features,fnames,target_num=10):
    model = sklearn.linear_model.LinearRegression()
    rfe = RFE(model,target_num)
    fit = rfe.fit(features,scores)
    fi = np.array(fit.support_)

    #print('ranking of features:')
    #for cnt,ii in enumerate(fit.ranking_):
    #    print('no. {}: {}'.format(cnt,fnames[ii]))

    return features[:,fi], fnames[fi]

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
        if len(variables) <= target_num:# or max(vif) < threshold:
            break
        else:
            variables = np.delete(variables,maxloc)

    #print('Remaining variables ({}):'.format(len(variables)))
    #for nn in variables:
        #print(fnames[nn])
    return features[:,variables], fnames[variables]

def printBaselineResults(features,names,scores):
    self_prevs = features[:,[nn=='self_prev' for nn in names]]
    all_means = features[:,[nn=='all_mean' for nn in names]]

    sp_error = mean_squared_error(self_prevs,scores)
    am_error = mean_squared_error(all_means,scores)

    print('\n=====BASELINES=====')
    print('previous score baseline error: {}'.format(sp_error))
    print('all previous mean baseline error: {}'.format(am_error))
    print()

def addRebuttalScoreFeature(features,names):
    self_prevs = features[:,[nn=='all_max' for nn in names]]
    added = [max(ss[0],4.) for ss in self_prevs]
    nfeatures = np.append(np.array(added).reshape(-1,1),features,axis=1)
    nnames = np.append('rebuttal_score',names)
    return nfeatures,nnames


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_num',action="store",dest="fn",nargs='?',default=-1)
    parser.add_argument("--classifier",action="store",dest="clf",default="lin-reg",nargs='?',help="gp|lin-reg|forest|tree")
    parser.add_argument("--feature-selector",action="store",dest="fs",default="vif",nargs='?',help='pca or vif')
    parser.add_argument("--feature-set",action="store",dest="fst",
                        default="opinion-politeness-specificity-length-convincingness-similarity",nargs='?')
                        #default="opinion",nargs='?')
    parser.add_argument("--data",action="store",dest="data",
                        default="./AggregateFeatures/borderline_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv",nargs='?')
    #parser.set_defaults(conll=False)

    parsed_args = parser.parse_args(sys.argv[1:])

    print(":::",parsed_args.fn,parsed_args.clf,parsed_args.fs,parsed_args.fst)

    cv = 50
    feature_num = int(parsed_args.fn)
    gp_type = parsed_args.clf
    feature_selector = parsed_args.fs
    feature_set_name = parsed_args.fst
    data_path = parsed_args.data

    ### read features and scores
    length_features = [1]
    opinion_features = list(range(2,28))
    #opinion_features = [2,4] ### only use self_prev and other_mean, all
    #opinion_features = [2,18] ### only use self_prev and all_mean, borderline
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

    data = pd.read_csv(data_path,usecols=feature_set+out_feature)
    print("-->",data_path,data.columns.values)
    data.fillna(0.)

    ### read features and scores
    feature_names= np.array(list(data.columns.values)[:-1])
    matrix = np.array(data.values)
    features = matrix[:,:-1]
    scores = matrix[:,-1]

    #features, feature_names = addRebuttalScoreFeature(features,feature_names)
    ### simple model
    #wanted_features = ['self_prev','all_mean','cvc_std','rev_resp_embd_sim','cvc_max','spec_max','plt_min','cvc_median', 'cvc_mean','plt_std','log_resp_length'] #all cases, significant features
    #wanted_features = ['self_prev','all_mean','log_resp_length','cvc_max','plt_min','rev_resp_embd_sim','plt_std','plt_max','spec_std','cvc_min','cvc_std','spec_max'] #borderline cases, significant features
    #wanted_features = ['self_prev','all_mean']
    #features = features[:,np.array([feature_names[ii] in wanted_features for ii in range(len(feature_names))])]
    #feature_names = np.array(wanted_features)
    ### simple model
    print('feature matrix size: {}'.format(features.shape))
    print(feature_names)
    features = StandardScaler().fit_transform(features)
    printCorrelation(scores,features,feature_names)
    #printBaselineResults(features,feature_names,scores)

    ### shuffle the order of the features and training examples
    indices = np.random.permutation(features.shape[0])
    features = features[indices]
    scores = scores[indices]

    ### shuffling the features
    indices = np.random.permutation(features.shape[1])
    feature_names = feature_names[indices]
    features = features[:,indices]

    #print(feature_names,features,":::")

    #print(features.shape,feature_names,len(feature_names),feature_num); sys.exit(1)
    print("-->",feature_num,"<--")
    if feature_selector == 'vif' and features.shape[1] > 1:
        features, feature_names = selectByVIF(features, feature_names, feature_num)
    elif feature_selector == 'pca' and features.shape[1] > 1:
        features, feature_names = selectByPCA(features,feature_names,feature_num)

    #print(features.shape,scores.shape,"<--0")
    F, pvalue = f_regression(features,scores)
    #print('\np-values: {}\n'.format(pvalue))

    regr = sklearn.linear_model.LinearRegression()
    #print(features.shape,scores.shape,"<--1")
    regr.fit(features,scores)
    pred = regr.predict(features)
    #print('coefficient: {}'.format(regr.coef_))
    for i in range(len(regr.coef_)):
        pval = pvalue[i]
        if pval<0.01: star="***"
        elif pval<0.05: star="**"
        elif pval<0.1: star="*"
        else: star=""
        if feature_names is not None:
            print(feature_names[i],"\t","%.3f"%regr.coef_[i],"\t","%.3f"%pvalue[i],star)
        else:
            print('feature {}'.format(i),"\t","%.3f"%regr.coef_[i],"\t","%.3f"%pvalue[i],star)

    print('mean squared error : {}, variance score : {}'.format(mean_squared_error(pred,scores),r2_score(pred,scores)))

    ### remove insignificant features
    #features = features[:,np.array([pp < 0.1 for pp in pvalue])]
    #feature_names = feature_names[np.array([pp < 0.1 for pp in pvalue])]

    if gp_type == 'gp':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    elif gp_type == 'svr_rbf':
        gp = SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif gp_type == 'svr_lin':
        gp = SVR(kernel='linear', C=1e3)
    elif gp_type == 'svr_poly':
        gp = SVR(kernel='poly', C=1e3, degree=2)
    elif gp_type == 'forest':
        gp = RandomForestRegressor(max_depth=6, random_state=0)
    elif gp_type == 'tree':
        gp = DecisionTreeRegressor(random_state=0)
    else:
        gp = sklearn.linear_model.LinearRegression()

    scores_cv = cross_val_score(gp,features,scores,cv=cv,scoring='neg_mean_squared_error')
    print('feature num: {}'.format(features.shape[1]))
    print('alg {}, {}-fold cv: mean {}, std {}'.format(gp_type,cv,scores_cv.mean(),scores_cv.std()))

