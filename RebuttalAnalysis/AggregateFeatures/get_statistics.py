import pandas as pd
import numpy as np

if __name__ == '__main__':
    all_path = 'all_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv'
    bd_path = 'borderline_score_cleanedPlt_cleanedSpc_cleanedCvc_sim_respLogLen.csv'
    all_data = pd.read_csv(all_path)
    bd_data = pd.read_csv(bd_path)

    ### get politeness
    bdp = bd_data.loc[:,'plt_max'].tolist()
    print('borderline plt max: min {}, max {}, mean {}, median {}, std {}'.format(
        np.min(bdp), np.max(bdp), np.mean(bdp), np.median(bdp), np.std(bdp)
    ))
    allp = all_data.loc[:,'plt_max'].tolist()
    print('all plt max: min {}, max {}, mean {}, median {}, std {}'.format(
        np.min(allp), np.max(allp), np.mean(allp), np.median(allp), np.std(allp)
    ))

