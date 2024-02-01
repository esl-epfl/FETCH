import pyedflib
import numpy as np
import pandas as pd
from utils.params import TUSZ_BENDR_channels, BENDR_channels
import json
import os


def get_annotation(filename, df):
    '''
    Args:
        filename:
        df:

    Returns:

    '''
    row = df.loc[filename.split('/')[-1]]
    fs = row['fs_FP1']
    onsets = [float(int(x[0]))/fs for x in row['onsets']]
    offsets = [float(int(x[0]))/fs for x in row['offsets']]
    length = row['length']/fs
    annotation = []
    if len(offsets) == 1 and offsets[0] == 0:  # Totally non-seizure
        annotation.append((0, length, 'T0'))
        return annotation
    else:
        if onsets[0] != 0:
            annotation.append((0, onsets[0], 'T0'))
        for i in range(len(onsets)-1):
            on_time = onsets[i]
            on_time_next = onsets[i+1]
            off_time = offsets[i]
            annotation.append((on_time, off_time, 'T1'))
            annotation.append((off_time, on_time_next, 'T0'))
        annotation.append((onsets[-1], offsets[-1], 'T1'))
        if offsets[-1] != length:
            annotation.append((offsets[-1],length, 'T0'))
        return annotation


def convert_signal(filename, df, save_path=None):
    global min_dataset
    f = pyedflib.EdfReader(filename + ".edf")
    signal_labels = f.getSignalLabels()
    valid_channels = []
    for idx, lbl in enumerate(signal_labels):
        if lbl in TUSZ_BENDR_channels.keys():
            valid_channels.append(idx)
    num_channels = len(valid_channels)
    assert num_channels != 0, "Number of valid channels is zero!"

    if save_path is None:
        save_path = filename
    f_w = pyedflib.EdfWriter(save_path + "_annotated.edf", num_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS)
    data_list = []
    for write_ch, ch in enumerate(valid_channels):
        f_w.setDigitalMaximum(write_ch, f.getDigitalMaximum(ch))
        f_w.setDigitalMinimum(write_ch, f.getDigitalMinimum(ch))
        f_w.setPhysicalMaximum(write_ch, f.getPhysicalMaximum(ch))
        f_w.setPhysicalMinimum(write_ch, f.getPhysicalMinimum(ch))
        # f_w.setLabel(write_ch, BENDR_channels[write_ch])
        f_w.setLabel(write_ch, TUSZ_BENDR_channels[f.getLabel(ch)])
        f_w.setSamplefrequency(write_ch, f.getSampleFrequency(ch))
        data_list.append(f.readSignal(ch))

    annotation = get_annotation(filename, df)
    for annot in annotation:
        f_w.writeAnnotation(annot[0], annot[1], annot[2])
    f_w.writeSamples(data_list)

    f.close()
    f_w.close()
    return np.min(data_list) ,  np.max(data_list)


def read_signal(filename):
    f = pyedflib.EdfReader(filename)
    print(f.readSignal(0).shape, np.mean(f.readSignal(0)))
    print(f.readSignal(10).shape, np.min(f.readSignal(10)))
    print(f.getLabel(0))
    print(f.getHeader())
    print(f.getSignalLabels())
    print(f.getSampleFrequencies())
    print(f.readAnnotations())
    f.close()


def get_full_path(x):
    mode, tcp, pat_group, pat_s_t = x.split('/')
    pat, s, _ = pat_s_t.split('_')

    s_fullname = [filename for filename in os.listdir("/home/amirshah/EPFL/BENDR/data/TUSZ/edf/{}/{}/{}/{}/".format(mode, tcp, pat_group, pat)) if
     filename.startswith(s)]
    assert len(s_fullname) == 1, "Error {}".format(s_fullname)
    return "{}/{}/{}/{}/{}/{}".format(mode, tcp, pat_group, pat, s_fullname[0], pat_s_t)


if __name__ == '__main__':
    df = pd.read_csv('../../input/TUSZ/TUSZ_labels.csv')
    df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df = df.set_index('file_name_edf')
    pd.set_option('max_columns', None)
    print(df[df['offsets'].apply(lambda x: x[0][0] == 0)].sample(n=5))
    minimum = 0
    maximum = 0

    df['full_path'] = df['file_name'].apply(get_full_path)
    seizure_df = df[df['offsets'].apply(lambda x: x[0][0] == 0)]
    for mode in ['trainSet', 'devSet', 'testSet']:
        seizure_files = seizure_df[seizure_df['mode'] == mode]['full_path'].tolist()
        os.mkdir(os.path.join('../../input/TUSZ', mode))
        for sf in seizure_files:
            save_path = '../../input/TUSZ/{}/{}'.format(mode, sf.split('/')[-1])
            min_dataset , max_dataset = convert_signal("/home/amirshah/EPFL/BENDR/data/TUSZ/edf/"+sf.split('.edf')[0],
                                                       df, save_path=save_path)
            if mode == 'trainSet':
                minimum = min(min_dataset, minimum)
                maximum = max(max_dataset, maximum)
    print("Min Max: ", minimum, maximum)

