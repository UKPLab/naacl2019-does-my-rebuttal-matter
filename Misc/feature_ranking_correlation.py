from scipy.stats import pearsonr

def getFeaturePvalues(fpath):
    ff = open(fpath,'r')
    features = []
    pvalues = []
    for line in ff.readlines():
        feature = line.split('    ')[0]
        pvalue = line.split('    ')[2].split('*')[0]
        features.append(feature)
        pvalues.append(float(pvalue))
    ff.close()

    return features,pvalues


if __name__ == '__main__':
    lf, lp = getFeaturePvalues('./feature_rankings/borderline_linear.txt')
    ff, fp = getFeaturePvalues('./feature_rankings/borderline_forest.txt')
    tf, tp = getFeaturePvalues('./feature_rankings/borderline_tree.txt')

    ### get common features
    cfeatures = set(lf).intersection(set(ff)).intersection(set(tf))
    print(cfeatures)
    olp = [lp[lf.index(cf)] for cf in cfeatures]
    ofp = [fp[ff.index(cf)] for cf in cfeatures]
    otp = [tp[tf.index(cf)] for cf in cfeatures]

    print('correlation between linear and forest: {}'.format(
        pearsonr(olp,ofp)
    ))
    print('correlation between tree and forest: {}'.format(
        pearsonr(otp,ofp)
    ))
    print('correlation between linear and tree: {}'.format(
        pearsonr(olp,otp)
    ))


