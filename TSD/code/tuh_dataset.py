import os
import argparse
import pickle
from multiprocessing import Pool
from random import shuffle
import math
import numpy as np
import random
import pandas as pd
import pyedflib
from scipy.signal import stft, resample
from scipy.signal import filtfilt, butter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--sample_rate', type=int, default=256)
parser.add_argument('--data_directory', type=str, default='/home/amirshah/EPFL/TUSZv2')
parser.add_argument('--save_directory', type=str,
                    default='/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess')
parser.add_argument('--label_type', type=str, default='csv_bi')
parser.add_argument('--cpu_num', type=int, default=32)
parser.add_argument('--data_type', type=str, default='eval', choices=['train', 'eval', 'dev'])
parser.add_argument('--task_type', type=str, default='binary', choices=['binary'])
parser.add_argument('--slice_length', type=int, default=12)
parser.add_argument('--eeg_type', type=str, default='stft', choices=['original', 'bipolar', 'stft'])
parser.add_argument('--selected_channels', type=int, default=-1)
args = parser.parse_args()

GLOBAL_INFO = {}

channels_groups = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 2, 4, 6, 12, 14, 16, 18],
    [1, 3, 5, 7, 13, 15, 17, 19],
    [0, 1, 12, 13, 14, 15, 10, 11],
    [6, 7, 10, 11, 16, 17, 18, 19],
    [0, 1, 4, 5, 14, 15, 18, 19],
    [0, 1, 2, 3, 12, 13, 14, 15],
    [1, 3, 4, 7, 8, 12, 13, 14], # RANDOM
    [0, 2, 3, 5, 14, 16, 17, 19], # RANDOM
    [0, 3, 4, 6, 7, 9, 17, 19], # RANDOM
]
TUSZv2_info_df = pd.read_json('../../input/TUSZv2_info.json')


def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extensions = info.get('extensions')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in extensions:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    return searched_list


def bipolar_signals_func(signals, selected_channels=None):
    """
    The function accepts a list of signals as input and returns a list of 20 bipolar signals, where 8 randomly selected
     elements are computed from the input signals, and the remaining 12 elements are NumPy zeros with the same shape as
      the input signals. The order of the original bipolar signals is preserved, ensuring that the position of the
       non-zero elements in the output list corresponds to their position in the original bipolar signals computation.

    Args:
        a list of input signals

    Returns:
        the computed bipolar signals list with randomly selected elements and the remaining elements as zeros,
        while maintaining the order of the original bipolar signals.

    """
    all_bipolar_signals = [
        signals[0] - signals[4],   # fp1-f7
        signals[1] - signals[5],   # fp2-f8
        signals[4] - signals[9],   # f7-t3
        signals[5] - signals[10],  # f8-t4
        signals[9] - signals[15],  # t3-t5
        signals[10] - signals[16], # t4-t6
        signals[15] - signals[13], # t5-o1
        signals[16] - signals[14], # t6-o2
        signals[9] - signals[6],   # t3-c3
        signals[7] - signals[10],  # c4-t4
        signals[6] - signals[8],   # c3-cz
        signals[8] - signals[7],   # cz-c4
        signals[0] - signals[2],   # fp1-f3
        signals[1] - signals[3],   # fp2-f4
        signals[2] - signals[6],   # f3-c3
        signals[3] - signals[7],   # f4-c4
        signals[6] - signals[11],  # c3-p3
        signals[7] - signals[12],  # c4-p4
        signals[11] - signals[13], # p3-o1
        signals[12] - signals[14]  # p4-o2
    ]

    if selected_channels is None or selected_channels == -1:
        indices = list(range(len(all_bipolar_signals)))
        random.shuffle(indices)
        selected_indices = set(indices[:8])
    else:
        selected_indices = channels_groups[selected_channels]
    bipolar_signals = [all_bipolar_signals[i] if i in selected_indices else
                       np.zeros_like(signals[0]) for i in range(len(all_bipolar_signals))]

    return bipolar_signals


def spectrogram_unfold_feature(signals):
    nperseg = 250
    noverlap = 50
    freq_resolution = 2
    nfft = args.sample_rate * freq_resolution
    freqs, times, spec = stft(signals, fs=args.sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                              boundary=None, padded=False)

    spec = spec[:, :spec.shape[1] - 1, :]
    spec = np.reshape(spec, (-1, spec.shape[2]))
    amp = (np.log(np.abs(spec) + 1e-10)).astype(np.float32)

    return freqs, times, amp


class TUHDataset(Dataset):
    def __init__(self, file_list, transform=None, selected_channels=None):
        self.file_list = file_list
        self.file_length = len(self.file_list)
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data_pkl = pickle.load(f)
            signals = np.asarray(data_pkl['signals'])
            if signals.shape != (20, 3072):
                print("Error in shape: ", signals.shape)

            if args.eeg_type == 'stft':
                f, t, signals = spectrogram_unfold_feature(signals)
                # print(signals.shape)
                # exit()

            signals = self.transform(signals)
            label = data_pkl['label']
            label = 0. if label == "bckg" else 1.
        return signals, label


def get_data_loader(batch_size):
    file_dir = {'train': os.path.join(args.save_directory, 'task-binary_datatype-train'),
                'val': os.path.join(args.save_directory, 'task-binary_datatype-eval'),
                'test': os.path.join(args.save_directory, 'task-binary_datatype-dev')}
    file_lists = {'train': {'bckg': [], 'seiz': []}, 'val': {'bckg': [], 'seiz': []}, 'test': {'bckg': [], 'seiz': []}}

    for dirname in file_dir.keys():
        filenames = os.listdir(file_dir[dirname])
        for filename in filenames:
            if 'bckg' in filename:
                file_lists[dirname]['bckg'].append(os.path.join(file_dir[dirname], filename))
            elif 'seiz' in filename:
                file_lists[dirname]['seiz'].append(os.path.join(file_dir[dirname], filename))
            else:
                print('------------------------  error  ------------------------')
                exit(-1)

    print('--------------------  file_lists  --------------------')
    for dirname in file_lists.keys():
        print('--------------------  {}'.format(dirname))
        for classname in file_lists[dirname].keys():
            print('{} num: {}'.format(classname, len(file_lists[dirname][classname])))

    train_data = file_lists['train']['bckg'] + file_lists['train']['seiz'] * \
                 int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
    shuffle(train_data)
    print('len(train_data): {}'.format(len(train_data)))

    val_data = file_lists['val']['bckg'] + file_lists['val']['seiz']
    test_data = file_lists['test']['bckg'] + file_lists['test']['seiz']
    print('len(val_data): {}'.format(len(val_data)))
    print('len(test_data): {}'.format(len(test_data)))

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = TUHDataset(train_data, transform=train_transforms, selected_channels=args.selected_channels)
    val_data = TUHDataset(val_data, transform=val_transforms, selected_channels=args.selected_channels)
    test_data = TUHDataset(test_data, transform=test_transforms, selected_channels=args.selected_channels)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_data, batch_size=math.ceil(len(val_data) / 50), shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=math.ceil(len(test_data) / 50), shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def _get_sample_frequency(signal_header):
    # Temporary conditional assignment while we deprecate 'sample_rate' as a channel attribute
    # in favor of 'sample_frequency', supporting the use of either to give
    # users time to switch to the new interface.
    return (signal_header['sample_rate']
            if signal_header.get('sample_frequency') is None
            else signal_header['sample_frequency'])


def read_edf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=False):
    """
    Convenience function for reading EDF+/BDF data with pyedflib.
    Will load the edf and return the signals, the headers of the signals
    and the header of the EDF. If all signals have the same sample frequency
    will return a numpy array, else a list with the individual signals
    Parameters
    ----------
    edf_file : str
        link to an edf file.
    ch_nrs : list of int, optional
        The indices of the channels to read. The default is None.
    ch_names : list of str, optional
        The names of channels to read. The default is None.
    digital : bool, optional
        will return the signals as digital values (ADC). The default is False.
    verbose : bool, optional
        Print progress bar while loading or not. The default is False.
    Returns
    -------
    signals : np.ndarray or list
        the signals of the chosen channels contained in the EDF.
    signal_headers : list
        one signal header for each channel in the EDF.
    header : dict
        the main header of the EDF file containing meta information.
    """
    assert (ch_nrs is None) or (ch_names is None), \
        'names xor numbers should be supplied'
    if ch_nrs is not None and not isinstance(ch_nrs, list): ch_nrs = [ch_nrs]
    if ch_names is not None and \
            not isinstance(ch_names, list): ch_names = [ch_names]

    with pyedflib.EdfReader(edf_file) as f:
        # see which channels we want to load
        available_chs = [ch.upper() for ch in f.getSignalLabels()]
        n_chrs = f.signals_in_file

        # find out which number corresponds to which channel
        if ch_names is not None:
            ch_nrs = []
            for ch in ch_names:
                if not ch.upper() in available_chs:
                    print('will be ignored.')
                else:
                    ch_nrs.append(available_chs.index(ch.upper()))

        # if there ch_nrs is not given, load all channels

        if ch_nrs is None:  # no numbers means we load all
            ch_nrs = range(n_chrs)

        # convert negative numbers into positives
        ch_nrs = [n_chrs + ch if ch < 0 else ch for ch in ch_nrs]

        # load headers, signal information and
        header = f.getHeader()
        signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]

        # add annotations to header
        annotations = f.readAnnotations()
        annotations = [[s, d, a] for s, d, a in zip(*annotations)]
        header['annotations'] = annotations

        signals = []
        for i, c in enumerate(tqdm(ch_nrs, desc='Reading Channels',
                                   disable=not verbose)):
            signal = f.readSignal(c, digital=digital)
            signals.append(signal)

        # we can only return a np.array if all signals have the same samplefreq
        sfreqs = [_get_sample_frequency(shead) for shead in signal_headers]
        all_sfreq_same = sfreqs[1:] == sfreqs[:-1]
        if all_sfreq_same:
            dtype = np.int32 if digital else float
            signals = np.array(signals, dtype=dtype)

    assert len(signals) == len(signal_headers), 'Something went wrong, lengths' \
                                                ' of headers is not length of signals'
    del f
    return signals, signal_headers, header


def generate_lead_wise_data(edf_file):
    filename = edf_file.split('/')[-1].split('.edf')[0]
    signals, signal_headers, header = read_edf(edf_file)
    file_info = TUSZv2_info_df.loc[filename]
    fs = file_info['sampling_frequency']
    length = file_info['length']
    labels = file_info['labels']
    num_target_samples = length * GLOBAL_INFO['sample_rate']
    signal_list = []
    disease_labels = {0: 'bckg', 1: 'seiz'}
    for bipolar_channel in file_info['bipolar_montage']:
        x = bipolar_channel[0]
        y = bipolar_channel[1]

        if x == -1 or y == -1:
            signal_list.append(np.zeros(num_target_samples))
            continue

        bipolar_signal = signals[x] - signals[y]
        # Define the band-pass filter parameters
        lowcut = 0.1  # Lower cutoff frequency in Hz
        highcut = 80  # Upper cutoff frequency in Hz
        order = 4  # Filter order (adjust as needed)

        # Calculate the normalized cutoff frequencies
        nyquist_freq = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq

        # Design and apply the band-pass filter
        b, a = butter(order, [low, high], btype='band')
        bipolar_signal_filtered = filtfilt(b, a, bipolar_signal)

        if fs != GLOBAL_INFO['sample_rate']:
            bipolar_signal_resampled = resample(bipolar_signal_filtered, num_target_samples)
            signal_list.append(bipolar_signal_resampled)
        else:
            signal_list.append(bipolar_signal_filtered)

    signal_list_ordered = np.asarray(signal_list)

    for i, label in enumerate(labels):
        slice_eeg = signal_list_ordered[:,
                    int(i * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']:
                    int((i + 1) * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']]

        with open("{}/{}_label_{}_index_{}.pkl".format(GLOBAL_INFO['save_directory'],
                                                       edf_file.split('/')[-1].split('.')[0],
                                                       disease_labels[label], i), 'wb') as f:
            pickle.dump({'signals': slice_eeg, 'patient id': edf_file.split('/')[-1].split('.')[0].split('_')[0],
                         'label': disease_labels[label]}, f)


def run_multi_process(f, l: list, n_processes=1):
    n_processes = min(n_processes, len(l))
    print('processes num: {}'.format(n_processes))

    results = []
    pool = Pool(processes=n_processes)
    for r in tqdm(pool.imap_unordered(f, l), total=len(l), ncols=75):
        results.append(r)

    pool.close()
    pool.join()

    return results


def main(args):
    channel_list = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG C3', 'EEG C4', 'EEG CZ',
                    'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    save_directory = "{}/task-{}_datatype-{}".format(args.save_directory, args.task_type, args.data_type)
    if os.path.isdir(save_directory):
        os.system("rm -r {}".format(save_directory))
    os.system("mkdir -p {}".format(save_directory))

    data_directory = "{}/edf/{}".format(args.data_directory, args.data_type)

    if args.task_type == "binary":
        disease_labels = {'bckg': 0, 'seiz': 1}
    else:
        exit(-1)

    edf_list = search_walk({'path': data_directory, 'extensions': [".edf", ".EDF"]})

    GLOBAL_INFO['channel_list'] = channel_list
    GLOBAL_INFO['disease_labels'] = disease_labels
    GLOBAL_INFO['save_directory'] = save_directory
    GLOBAL_INFO['label_type'] = args.label_type
    GLOBAL_INFO['sample_rate'] = args.sample_rate
    GLOBAL_INFO['slice_length'] = args.slice_length
    # GLOBAL_INFO['disease_type'] = args.disease_type

    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_INFO:
        print("{}: {}".format(i, GLOBAL_INFO[i]))
    with open(data_directory + '/preprocess_info.pickle', 'wb') as pkl:
        pickle.dump(GLOBAL_INFO, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    # for edf_file in tqdm(edf_list[:4]):
    #     generate_lead_wise_data(edf_file)
    run_multi_process(generate_lead_wise_data, edf_list, n_processes=6)


if __name__ == '__main__':
    main(args)
