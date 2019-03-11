import pandas as pd
from resources import BASE_PATH

def getSubInfo(pids):
    data = pd.read_csv('{}/../data/MetaData/Submission_Information.csv'.format(BASE_PATH))
    data = data.loc[:,['Submission ID','Track','Submission Type','Acceptance Status']]
    accs = []
    types = []
    tracks = []
    for pid in pids:
        accs.append(data[data['Submission ID']==pid]['Acceptance Status'].values[0])
        types.append(data[data['Submission ID']==pid]['Submission Type'].values[0])
        tracks.append(data[data['Submission ID']==pid]['Track'].values[0])
    return accs, types, tracks

def getResponses(pids):
    data = pd.read_csv('{}/../consent_data/Author_Response_Information.csv'.format(BASE_PATH))
    resps = []
    for pid in pids:
        entry = data[data['Submission ID']==pid]
        if not entry.empty:
            resps.append(entry['Responses To Individual Reviews'].values[0])
        else:
            resps.append('no_permission')
    return resps, data['Submission ID'].tolist()


if __name__ == '__main__':
    before = pd.read_csv('{}/../consent_data/Before_Rebuttal_Reviews.csv'.format(BASE_PATH))
    print('before reviews size: {}'.format(before.shape[0]))
    bpids = set(before['Submission ID'].tolist())
    print('before pid num: {}'.format(len(bpids)))
    breviewers = set(before['Reviewer Username'].tolist())
    print('before reviewer num: {}'.format(len(breviewers)))

    after = pd.read_csv('{}/../consent_data/After_Rebuttal_Reviews.csv'.format(BASE_PATH))
    print('after reviews size: {}'.format(after.shape[0]))
    apids = set(after['Submission ID'].tolist())
    print('after pid num: {}'.format(len(apids)))
    areviewers = set(after['Reviewer Username'].tolist())
    print('after reviewer num: {}'.format(len(areviewers)))

    print('both after and before pids: {}'.format(len(apids.intersection(bpids))))

    '''
    accs, types, tracks = getSubInfo(apids)
    #new_data = pd.DataFrame(data={'PID':list(apids),'Track':tracks, 'Submission Type':types, 'AcceptanceStatus':accs})
    #new_data.to_csv('Submission_Information.csv')

    acpt = [a for a in accs if 'Accept' in a]
    rejc = [a for a in accs if 'Reject' in a]
    print('acc num: {}, rejc num: {}'.format(len(acpt),len(rejc)))


    long = [a for a in types if 'Long' in a]
    short = [a for a in types if 'Short' in a]
    print('long num: {}, short num: {}'.format(len(long),len(short)))

    long_acc = 0
    short_acc = 0
    for ii in range(len(accs)):
        if 'Accept' in accs[ii] and 'Long' in types[ii]:
            long_acc += 1
        elif 'Accept' in accs[ii] and 'Short' in types[ii]:
            short_acc += 1
    print('long acc {}, short acc {}'.format(long_acc,short_acc))
    '''

    resp_list, resp_pids = getResponses(apids)
    print('no permissions num : {}'.format(len([rr for rr in resp_list if rr=='no_permission'])))
    print(set(resp_pids)-apids)
    print('empty response num : {}'.format(len([rr for rr in resp_list if repr(rr).lower()=='nan'])))











