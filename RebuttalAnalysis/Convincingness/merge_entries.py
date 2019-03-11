import pandas as pd

### merge sentences from Increase, Decrease and NoChange into one file
### purpose: when computing the convincingness scores, features will be extracted from all sentences
### sample use: python merge_entries.py

def mergeEntries(names):
    index_output = ''
    sent_output = []

    for nn in names:
        ff = open('../TokenisedResponses/{}.index'.format(nn),'r')
        for ll in ff.readlines():
            index_output += '{}\t{}\n'.format(ll.strip(),nn)
        ff.close()

        ff = open('../TokenisedResponses/{}.sent'.format(nn),'r')
        for ll in ff.readlines():
            entry = [0.0,ll.strip(),'dummyTurker']
            sent_output.append(entry)
        ff.close()

    ff = open('all.index','w')
    ff.write(index_output)
    ff.close()

    data = pd.DataFrame(sent_output,columns=['rank','argument','turkID'])
    data.to_csv('all.argument','ß')

if __name__ == '__main__':
    names = ['Increase','Decrease','NoChange']
    #mergeEntries(names)

    all_df = None
    for nn in names:
        data = pd.read_csv('{}_cvc.csv'.format(nn))
        if all_df is None:
            all_df = data
        else:
            all_df = all_df.append(data,sort=False)

    all_df.to_csv('all_cvc.csv')
