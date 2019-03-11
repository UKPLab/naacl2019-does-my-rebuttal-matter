import pandas as pd

def findEmptyResp():
    names = ['Increase','Decrease','NoChange']
    for nn in names:
        pids = []
        bscores = []
        ascores = []
        data = pd.read_csv('{}.csv'.format(nn))
        for _,entry in data.iterrows():
            rr = entry['AuthorResponse']
            if repr(rr) == 'nan':
                bscores.append(float(entry['BeforeScore']))
                ascores.append(float(entry['AfterScore']))
                pids.append(int(entry['PID']))

        assert len(bscores) == len(ascores)
        print('In {} responses, {}/{} are empty'.format(nn,len(bscores),data.shape[0]))
        if len(bscores) > 0:
            print('pids:\t\t{}'.format(pids))
            print('before scores:\t\t{}'.format(bscores))
            print('after scores:\t\t{}'.format(ascores))


if __name__ == '__main__':
    #findEmptyResp()

    ### 328 and 135 are two papers that reply to some of its reviews and ignore other reviews (on purpose)
    data = pd.read_csv('{}.csv'.format('NoChange'))
    for _,entry in data.iterrows():
        if int(entry['PID']) in [328,135]:
            print('pid {}, review {}, rebuttal {}'.format(entry['PID'],entry['BeforeReview'],entry['AuthorResponse']))
