import numpy as np
import pandas as pd
import scipy.io
from params import pat_file_list
import pandas

dir_seizure_files = '../../input/Epilepsiae_info/seizure_files'
df = pd.DataFrame(columns=['patient', 'file_name', 'onsets', 'offsets', 'length', 'channels'])
for filename in pat_file_list:
    print(filename)

    data = scipy.io.loadmat('{}/{}'.format(dir_seizure_files, filename))
    new_row = {'patient': data['patient'][0], 'file_name': filename, 'onsets': data['Onsets'],'offsets': data['Offsets'],
               'length':data['Signals'].shape[1], 'channels': data['EEG_channels']}
    # new_row = {'patient': 0, 'file_name': filename, 'onsets': 1,'offsets': 1}
    df = df.append(new_row, ignore_index=True)
df.to_csv('../../output/epilepsiae_labels.csv')
