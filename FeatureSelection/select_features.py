import sys, operator

def getCorrByFeature(data,feature):
    dic = {}
    for entry in data:
        if feature in entry[0]:
            dic[entry[0]] = float(entry[1])

    sorted_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    for entry in sorted_dic:
        print('{}\t{}'.format(entry[0],entry[1]))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        fpath = sys.argv[1]
        fname = sys.argv[2]
    else:
        fpath = 'feature_correlation_borderline.txt'
        fname = 'self_prev'

    data = []
    ff = open(fpath,'r')
    for ii,line in enumerate(ff.readlines()):
        if ii == 0:
            continue
        if line.strip() == '':
            break
        data.append(line.split(':'))

    getCorrByFeature(data,fname)