from sklearn.metrics import cohen_kappa_score
import numpy as np

def getUsersKappa(anns):
    klist = []
    for ii in range(len(anns)-1):
        for jj in range(ii+1,len(anns)):
            agr = [1 for id in range(len(anns[ii])) if anns[ii][id]==anns[jj][id]]
            klist.append(np.sum(agr)*1./len(anns[ii]))
            print('kappa between user {} and {}: {}'.format(ii,jj,np.sum(agr)/len(anns[ii])))

    print('average over all {} users kappa: {}'.format(len(anns),np.mean(klist)))
    return klist

def aggregateByVoting(anns):
    result = []
    for ii in range(len(anns[0])):
        ll = [aa[ii] for aa in anns]
        agg = max(set(ll), key = ll.count)
        result.append(agg)
    return result

if __name__ == '__main__':
    ### politeness
    pol1 = [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    pol2 = [1,0,0,0,0,1,0,0,0,0,0,0,0,1,1]
    pol3 = [1,0,0,0,0,1,0,0,0,0,0,1,0,1,0]
    pols = [1,0,0,0,0,1,0,0,0,0,0,1,0,1,1]
    print('\n---politeness between users---')
    getUsersKappa([pol1,pol2,pol3])
    print('---politeness between aggregated user and score---')
    getUsersKappa([aggregateByVoting([pol1,pol2,pol3]),pols])

    ### specificity
    spc1 = [0,0,0,1,0,1,1,1,0,1,0,0,1,0,1]
    spc2 = [0,0,0,1,1,1,1,1,0,1,0,1,1,0,1]
    spc3 = [0,1,0,1,1,1,1,1,0,1,0,1,1,0,1]
    spcs = [0,1,0,1,1,1,1,1,0,1,0,1,1,0,1]
    print('\n---specificity between users---')
    getUsersKappa([spc1,spc2,spc3])
    print('---specificity between aggregated user and score---')
    getUsersKappa([aggregateByVoting([spc1,spc2,spc3]),spcs])

    ### convincingness
    cvc1 = [1,0,0,0,0,0,0,0,1,1,0,0,1,0,0]
    cvc2 = [1,0,1,1,1,0,1,1,1,1,0,1,1,0,0]
    cvc3 = [1,1,0,0,0,0,0,0,1,0,0,0,1,0,0]
    cvcs = [1,0,1,0,1,0,1,0,1,0,0,1,1,0,0]
    print('\n---convincingness between users---')
    getUsersKappa([cvc1,cvc2,cvc3])
    print('---convincingness between aggregated user and score---')
    getUsersKappa([aggregateByVoting([cvc1,cvc2,cvc3]),cvcs])
