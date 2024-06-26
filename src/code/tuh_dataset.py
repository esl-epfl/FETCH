import json
import os
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import stft, resample
from scipy.signal import filtfilt, butter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
from parser_util import get_parser

torch.random.manual_seed(42)  # optional: for reproducibility

GLOBAL_INFO = {}

# channels_groups = [
#     [0, 1, 2, 3, 4, 5, 6, 7],
#     [0, 2, 4, 6, 12, 14, 16, 18],
#     [1, 3, 5, 7, 13, 15, 17, 19],
#     [0, 1, 12, 13, 14, 15, 10, 11],
#     [6, 7, 10, 11, 16, 17, 18, 19],
#     [0, 1, 4, 5, 14, 15, 18, 19],
#     [0, 1, 2, 3, 12, 13, 14, 15],
#     [1, 3, 4, 7, 8, 12, 13, 14], # RANDOM
#     [0, 2, 3, 5, 14, 16, 17, 19], # RANDOM
#     [0, 3, 4, 6, 7, 9, 17, 19], # RANDOM
# ]

TUSZv2_info_df = pd.read_json('../../input/TUSZv2_info.json')
args = get_parser().parse_args()


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
    def __init__(self, file_list, signals=None, labels= None, transform=None,
                 selected_channel_id=-1, masking=True, remove_not_used=False):
        self.file_list = file_list
        self.signals = signals
        self.labels = labels
        self.file_length = len(self.file_list) if not self.file_list is None else self.signals.shape[0]
        self.transform = transform
        self.masking = masking
        self.remove_not_used_channels = remove_not_used
        with open("../feasible_channels/feasible_20edges.json", 'r') as json_file:
            self.all_feasible_channel_combination = json.load(json_file)
        if selected_channel_id == -1:
            self.selected_channels = None
        else:
            self.selected_channels = self.all_feasible_channel_combination[selected_channel_id]

        print("Selected channels: ", self.selected_channels)

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        if self.file_list is None:
            if self.signals is None:
                raise ValueError("Both file list and signals are None!")
            signals = self.signals[idx].clone()
            label = self.labels[idx].copy()
        else:
            with open(self.file_list[idx], 'rb') as f:
                data_pkl = pickle.load(f)
                signals = torch.from_numpy(np.asarray(data_pkl['STFT']))
                label = data_pkl['label']
                label = 0. if label == "bckg" else 1.

        if self.remove_not_used_channels:
            present_channels = self.selected_channels
            signals = signals[present_channels]
            signals = torch.reshape(signals, (-1, signals.shape[2]))
            signals = signals.unsqueeze(0)

        elif self.masking:
            MASK = torch.ones(20, dtype=torch.bool)
            # Create a list of indices
            indices = torch.arange(20)

            if self.selected_channels is None:  # if the mask is not pre-assigned
                # Randomly shuffle the indices
                # TODO: get random permutation from the all_feasible_channel_combination
                indices = indices[torch.randperm(20)]

                # Select the first 8 indices and assign 0 to the corresponding MASK elements
                MASK[indices[:8]] = 0
            else:
                present_channels = self.selected_channels
                MASK[present_channels] = 0

            signals[MASK] = -1  # Set all elements corresponding to True in MASK to -1

            signals = torch.reshape(signals, (-1, signals.shape[2]))
            signals = signals.unsqueeze(0)

        return signals, label


class TUHDatasetValidation(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.file_length = len(self.file_list)
        self.transform = transform
        with open('../../TUSZv2/validation_file_mask8_dict.pkl', 'rb') as f:
            self.file_mask_dict = pickle.load(f)

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data_pkl = pickle.load(f)
            signals = np.asarray(data_pkl['STFT'])

            filename = self.file_list[idx].split('/')[-1]
            MASK = self.file_mask_dict[filename]

            signals[MASK] = -1  # Set all elements corresponding to True in MASK to -1

            signals = np.reshape(signals, (-1, signals.shape[2]))
            signals = self.transform(signals)

            label = data_pkl['label']
            label = 0. if label == "bckg" else 1.
        return signals, label


class TUHDatasetEvent(Dataset):
    def __init__(self, recording_list, transform=None):
        self.recording_list = recording_list
        self.transform = transform

    def __len__(self):
        return len(self.recording_list)

    def __getitem__(self, idx):
        filenames = self.recording_list[idx]
        recording_signals = []
        recording_labels = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                data_pkl = pickle.load(f)
                signals = np.asarray(data_pkl['signals'])
                if signals.shape != (20, 3072):
                    print("Error in shape: ", signals.shape)

                if args.eeg_type == 'stft':
                    f, t, signals = spectrogram_unfold_feature(signals)
                    # print(signals.shape)
                    # exit()

                signals = self.transform(signals)
                recording_signals.append(signals)
                label = data_pkl['label']
                label = 0. if label == "bckg" else 1.
                recording_labels.append(label)
        return np.vstack(recording_signals), np.vstack(recording_labels)


def separate_and_sort_filenames(filenames):
    parts_dict = {}
    for filename in filenames:
        key = filename.split('_label')[0]
        index = int(filename.split('index_')[-1].split('.pkl')[0])
        if key not in parts_dict:
            parts_dict[key] = []
        parts_dict[key].append((filename, index))

    sorted_lists = []
    for key, value in parts_dict.items():
        sorted_values = sorted(value, key=lambda x: x[1])
        sorted_list = [filename for filename, index in sorted_values]
        sorted_lists.append(sorted_list)

    return sorted_lists


def get_data(save_dir=args.save_directory, balanced_data=True, return_val_test_signal=False,
             return_train_signal= False):
    # Specify the output filename
    file_lists_filename = os.path.join(args.save_directory, "./file_lists.pkl")
    if not os.path.exists(file_lists_filename):
        file_dir = {'train': os.path.join(save_dir, 'task-binary_datatype-train_STFT'),
                    'val': os.path.join(save_dir, 'task-binary_datatype-dev_STFT'),
                    'test': os.path.join(save_dir, 'task-binary_datatype-eval_STFT')}
        file_lists = {'train': {'bckg': [], 'seiz': []}, 'val': {'bckg': [], 'seiz': []},
                      'test': {'bckg': [], 'seiz': []}}

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

        # Save file_lists using pickle
        with open(file_lists_filename, "wb") as pickle_file:
            pickle.dump(file_lists, pickle_file)

    else:
        # Save file_lists using pickle
        with open(file_lists_filename, "rb") as pickle_file:
            file_lists = pickle.load(pickle_file)

    print('--------------------  file_lists  --------------------')
    for dirname in file_lists.keys():
        print('--------------------  {}'.format(dirname))
        for classname in file_lists[dirname].keys():
            print('{} num: {}'.format(classname, len(file_lists[dirname][classname])))

    if balanced_data:
        train_data = file_lists['train']['bckg'] + file_lists['train']['seiz'] * \
                     int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
        seizure_labels = np.ones(len(file_lists['train']['seiz']) *
                                 int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
                                 , dtype=np.int32)
    else:
        train_data = file_lists['train']['bckg'] + file_lists['train']['seiz']
        seizure_labels = np.ones(len(file_lists['train']['seiz']), dtype=np.int32)

    non_seizure_labels = np.zeros(len(file_lists['train']['bckg']), dtype=np.int32)
    train_label = np.concatenate((non_seizure_labels, seizure_labels))
    print('len(train_data): {}'.format(len(train_data)))

    val_data = file_lists['val']['bckg'] + file_lists['val']['seiz']
    test_data = file_lists['test']['bckg'] + file_lists['test']['seiz']

    print('len(val_data): {}'.format(len(val_data)))
    print('len(test_data): {}'.format(len(test_data)))

    if return_train_signal and return_val_test_signal:
        val_label = np.concatenate((np.zeros(len(file_lists['val']['bckg'])),
                                    np.ones(len(file_lists['val']['seiz']))))
        test_label = np.concatenate((np.zeros(len(file_lists['test']['bckg'])),
                                     np.ones(len(file_lists['test']['seiz']))))

        train_signal = torch.zeros((len(train_data), 20, 160, 15), dtype=torch.float)
        validation_signal = torch.zeros((len(val_data), 20, 160, 15), dtype=torch.float)
        test_signal = torch.zeros((len(test_data), 20, 160, 15), dtype=torch.float)
        for input_signal, input_data_files in zip([train_signal, validation_signal, test_signal],
                                                  [train_data, val_data, test_data]):
            for idx in tqdm(range(len(input_data_files)), desc="Reading input files"):
                with open(input_data_files[idx], 'rb') as f:
                    data_pkl = pickle.load(f)
                    input_signal[idx, :, :, :] = torch.from_numpy(np.asarray(data_pkl['STFT']))

        return (train_data, val_data, test_data,
                train_signal, train_label,
                validation_signal, val_label,
                test_signal, test_label)  # TODO: make a dictionary or class for returning

    elif return_val_test_signal:
        val_label = np.concatenate((np.zeros(len(file_lists['val']['bckg'])),
                                    np.ones(len(file_lists['val']['seiz']))))
        test_label = np.concatenate((np.zeros(len(file_lists['test']['bckg'])),
                                     np.ones(len(file_lists['test']['seiz']))))

        validation_signal = torch.zeros((len(val_data), 20, 160, 15), dtype=torch.float)
        test_signal = torch.zeros((len(test_data), 20, 160, 15), dtype=torch.float)
        for input_signal, input_data_files in zip([validation_signal, test_signal], [val_data, test_data]):
            for idx in tqdm(range(len(input_data_files)), desc="Reading input files"):
                with open(input_data_files[idx], 'rb') as f:
                    data_pkl = pickle.load(f)
                    input_signal[idx, :, :, :] = torch.from_numpy(np.asarray(data_pkl['STFT']))

        return (train_data, val_data, test_data,
                None, train_label,
                validation_signal, val_label,
                test_signal, test_label)

    else:
        return train_data, val_data, test_data, None, train_label, None, None, None, None


def get_dataloader(train_data, val_data, test_data,
                   train_signal, train_label,
                   validation_signal, val_label, test_signal, test_label,
                   batch_size, event_base=False, random_mask=False,
                   return_dataset=False, masking=True, remove_not_used=False,
                   selected_channel_id = args.selected_channel_id, ):

    train_transforms = transforms.ToTensor()

    val_transforms = transforms.ToTensor()

    test_transforms = transforms.ToTensor()

    if random_mask:
        train_data = TUHDataset(train_data, transform=train_transforms, masking=masking)
        if event_base:
            val_data = TUHDatasetEvent(separate_and_sort_filenames(val_data), transform=val_transforms)
            test_data = TUHDatasetEvent(separate_and_sort_filenames(test_data), transform=test_transforms)
        else:
            val_data = TUHDatasetValidation(val_data, transform=val_transforms)
            test_data = TUHDataset(test_data, transform=test_transforms)
    else:
        # TODO: set selected channels based on json and the args.selected_channel_id
        if train_data is None:
            train_data = TUHDataset(None, signals=train_signal, labels=train_label,
                                    transform=train_transforms,
                                    selected_channel_id=selected_channel_id,
                                    masking=masking, remove_not_used=remove_not_used)
        else:
            train_data = TUHDataset(train_data, signals=None, labels=None, transform=train_transforms,
                                selected_channel_id=selected_channel_id,
                                masking=masking, remove_not_used=remove_not_used)
        if val_data is None:  # Using signals and labels to speedup. The drawback is that it occupies more memory in GPU
            val_data = TUHDataset(None, signals=validation_signal, labels=val_label, transform=val_transforms,
                                  selected_channel_id=selected_channel_id,
                                  masking=masking, remove_not_used=remove_not_used)
            test_data = TUHDataset(None, signals=test_signal, labels=test_label, transform=test_transforms,
                                   selected_channel_id=selected_channel_id,
                                   masking=masking, remove_not_used=remove_not_used)
        else:
            val_data = TUHDataset(val_data, signals= None, labels=None, transform=val_transforms, masking=masking,
                                  selected_channel_id=selected_channel_id,
                                  remove_not_used=remove_not_used)
            test_data = TUHDataset(test_data, signals=None, labels=None, transform=test_transforms, masking=masking,
                                   selected_channel_id=selected_channel_id,
                                   remove_not_used=remove_not_used)

    if return_dataset:
        return train_data, val_data, test_data

    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=6)

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


def generate_STFT(pickle_file):
    save_directory = "{}/task-{}_datatype-{}_STFT".format(args.save_directory, args.task_type, args.data_type)

    nperseg = 256
    noverlap = 64
    sampling_rate = 256
    freq_resolution = 2
    nfft = sampling_rate * freq_resolution
    cutoff_freq = 80

    with open(pickle_file, 'rb') as f:
        data_pkl = pickle.load(f)
        signals = np.asarray(data_pkl['signals'])
        if signals.shape != (20, 3072):
            print("Error in shape: ", signals.shape)

        freqs, times, spec = stft(signals, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                  boundary=None, padded=False)

        spec = spec[:, :cutoff_freq*freq_resolution, :]
        amp = (np.log(np.abs(spec) + 1e-10)).astype(np.float32)

        label = data_pkl['label']
        with open("{}/{}.pkl".format(save_directory, pickle_file.split('/')[-1].split('.')[0]), 'wb') as out_f:
            pickle.dump({'STFT': amp, 'label': label}, out_f)


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


def make_STFT(args):
    save_directory = "{}/task-{}_datatype-{}_STFT".format(args.save_directory, args.task_type, args.data_type)
    if os.path.isdir(save_directory):
        os.system("rm -r {}".format(save_directory))
    os.system("mkdir -p {}".format(save_directory))

    data_directory = "{}/task-{}_datatype-{}".format(args.save_directory, args.task_type, args.data_type)
    pickle_list = []
    for pickle_file in os.listdir(data_directory):
        if pickle_file.endswith(".pkl"):
            pickle_list.append(os.path.join(data_directory, pickle_file))

    run_multi_process(generate_STFT, pickle_list, n_processes=6)
    # for pickle_file in tqdm(pickle_list[:1]):
    #     generate_STFT(pickle_file)


def save_validation_inference():
    batch_size = 100
    _, val_loader, test_loader = get_data_loader(batch_size)
    validation_file_mask_dict = {}
    for idx, (data, label, file, mask) in enumerate(tqdm(val_loader, desc='Validation ')):
        file_mask_dict = dict(zip(file, mask))
        validation_file_mask_dict.update(file_mask_dict)

    # Save 'dict' to a file
    with open('validation_file_mask8_dict.pkl', 'wb') as f:
        pickle.dump(validation_file_mask_dict, f)


if __name__ == '__main__':
    # make_STFT(args)
    pass
