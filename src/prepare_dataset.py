import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io, scipy.integrate
from scipy.signal import resample
from scipy.signal import stft
import os
import pickle
# from eglass import calculateMLfeatures
from utils.params import pat_file_list, EEG_channels, EEG_channels_LE
import pyedflib
from tqdm import tqdm
import random

from scipy import signal
from os import listdir
from os.path import isfile, join
import sys
import json
import seaborn as sns

# fs = 250
TUSZ_folder = "train/03_tcp_ar_a"


def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0


def bandpower(f, Pxx, fmin, fmax):
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return scipy.integrate.trapz(Pxx[ind_min: ind_max + 1], f[ind_min: ind_max + 1])


def calculateOtherMLfeatures_oneCh(X, fs):
    numFeat = 5  # 4 from Sopic2018 and LL and meanAmpl
    lenSig = len(X)
    segLenIndx = int(4 * fs)  # length of EEG segments in samples
    slidWindStepIndx = int(1 * fs)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx + slidWindStepIndx, slidWindStepIndx).astype(int)

    featureValues = np.zeros((len(index), numFeat))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]
        if len(sig) != segLenIndx: print("Error in signal length!")
        f, Pxx = scipy.signal.periodogram(sig, fs=fs)
        p_delta = bandpower(f, Pxx, 0.5, 4)
        p_theta = bandpower(f, Pxx, 4, 8)
        p_alfa = bandpower(f, Pxx, 8, 13)
        p_beta = bandpower(f, Pxx, 13, 30)
        LL = np.mean(np.abs(np.diff(sig)))
        featureValues[i, :] = np.hstack((LL, p_delta, p_theta, p_alfa, p_beta))
    return featureValues


def calculateMovingAvrgMeanWithUndersampling_v2(data, winLen, winStep):
    lenSig = len(data)
    index = np.arange(0, lenSig - winLen + winStep, winStep)

    segmData = np.zeros(len(index))
    for i in range(len(index)):  # -1
        x = data[index[i]:index[i] + winLen]
        segmData[i] = np.mean(x)
    return (segmData)


def polygonal_approx(arr, epsilon):
    """
    Performs an optimized version of the Ramer-Douglas-Peucker algorithm assuming as an input
    an array of single values, considered consecutive points, and **taking into account only the
    vertical distances**.
    """

    def max_vdist(arr, first, last):
        """
        Obtains the distance and the index of the point in *arr* with maximum vertical distance to
        the line delimited by the first and last indices. Returns a tuple (dist, index).
        """
        if first == last:
            return (0.0, first)
        frg = arr[first:last + 1]
        leng = last - first + 1
        dist = np.abs(frg - np.interp(np.arange(leng), [0, leng - 1], [frg[0], frg[-1]]))
        idx = np.argmax(dist)
        return (dist[idx], first + idx)

    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    if len(arr) < 3:
        return arr
    result = set()
    stack = [(0, len(arr) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, idx = max_vdist(arr, first, last)
        if max_dist > epsilon:
            stack.extend([(first, idx), (idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))


def get_Eglass_features(allsigFilt, fs):
    # %%
    numCh = len(EEG_channels)
    num_feat = 12
    win_len = 4
    eps = 1e-6
    EPS_thresh_arr = [0.01, 0.04, 0.1, 0.4, 0.8]
    length = int(allsigFilt.shape[1] // fs) - win_len + 1

    valid_num_samples = (allsigFilt.shape[1] // fs) * fs

    scaleFactor = np.max(allsigFilt, axis=1, keepdims=True) - np.min(allsigFilt, axis=1, keepdims=True)
    allSigScaled = (allsigFilt - np.min(allsigFilt, axis=1, keepdims=True)) / (scaleFactor + eps)
    allSigScaled = allSigScaled * 2 - 1

    all_features = np.zeros((length, num_feat * numCh))  # Zero-crossing, band powers, LL, and scale_factor

    for ch in range(numCh):
        sigFiltScaled = allSigScaled[ch, :valid_num_samples]
        new_fs = 256
        new_num_samples = (allsigFilt.shape[1] // fs) * new_fs
        sigFiltScaledResampled = resample(sigFiltScaled, new_num_samples)

        featOther = calculateOtherMLfeatures_oneCh(np.copy(sigFiltScaledResampled), new_fs)
        all_features[:, ch * num_feat] = scaleFactor[ch] / 5000
        all_features[:, ch * num_feat + 1:ch * num_feat + 6] = featOther

        x = np.convolve(zero_crossings(sigFiltScaledResampled), np.ones(new_fs), mode='same')
        zeroCrossStandard = calculateMovingAvrgMeanWithUndersampling_v2(x, new_fs * 4, new_fs)
        all_features[:, ch * num_feat + 6] = zeroCrossStandard
        for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
            sigApprox = polygonal_approx(sigFiltScaledResampled, epsilon=EPSthr)
            sigApproxInterp = np.interp(np.arange(len(sigFiltScaledResampled)), sigApprox,
                                        sigFiltScaledResampled[sigApprox])
            x = np.convolve(zero_crossings(sigApproxInterp), np.ones(new_fs), mode='same')
            zeroCrossApprox = calculateMovingAvrgMeanWithUndersampling_v2(x, new_fs * 4, new_fs)
            all_features[:, ch * num_feat + 7 + EPSthrIndx] = zeroCrossApprox

    return all_features


def get_stft_features(all_signal_filtered, fs):
    length = int(all_signal_filtered.shape[1] // fs)
    num_ch = len(EEG_channels)
    frequency_stop = 50
    stft_time_len = 14
    frequency_resolution = 2

    valid_num_samples = (all_signal_filtered.shape[1] // fs) * fs
    new_fs = 250
    new_num_samples = (all_signal_filtered.shape[1] // fs) * new_fs
    number_of_12_seconds = int(new_num_samples // (12 * new_fs))
    all_features = np.zeros((number_of_12_seconds, num_ch, frequency_stop * frequency_resolution, stft_time_len))
    for ch in range(num_ch):
        single_channel_signal = all_signal_filtered[ch, :valid_num_samples]
        single_channel_signal = resample(single_channel_signal, new_num_samples)
        for i in range(number_of_12_seconds):
            f, t, Zxx = stft(single_channel_signal[i*12*new_fs: (i+1)*12*new_fs], fs=new_fs,
                             nfft=new_fs * frequency_resolution,
                             nperseg=new_fs, noverlap=50, padded=False, boundary=None)
            Zxx = Zxx[:frequency_stop * frequency_resolution, :]
            Zxx = np.reshape(Zxx, (-1, Zxx.shape[1]))
            all_features[i, ch, :, :] = (np.log(np.abs(Zxx) + 1e-10)).astype(np.float32)

    return all_features


def get_EEG_index(eeg_labels):
    indices = []
    for label, label_le in zip(EEG_channels, EEG_channels_LE):
        if label in eeg_labels:
            indices.append(eeg_labels.index(label))
        elif label_le in eeg_labels:
            indices.append(eeg_labels.index(label_le))
        else:
            indices.append(-1)
    return indices


def get_full_path(x):
    mode, tcp, pat_group, pat_s_t = x.split('/')
    pat, s, _ = pat_s_t.split('_')

    # s_fullname = [filename for filename in os.listdir("/scrap/users/amirshah/epilepsyTrans/TUSZ/edf/{}/{}/{}/{}/".format(mode, tcp, pat_group, pat)) if
    s_fullname = [filename for filename in os.listdir("/home/amirshah/EPFL/EpilepsyTransformer/input/TUSZ/edf/{}/{}/{}/{}/".format(mode, tcp, pat_group, pat)) if
     filename.startswith(s)]
    assert len(s_fullname) == 1, "Error {}".format(s_fullname)
    return "{}/{}/{}/{}/{}/{}".format(mode, tcp, pat_group, pat, s_fullname[0], pat_s_t)


def get_filename_list(pat_num):
    df = pd.read_csv('../input/TUSZ/TUSZ_labels.csv')
    df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df = df.set_index('file_name_edf')
    pd.set_option('max_columns', None)

    df['full_path'] = df['file_name'].apply(get_full_path)
    # df = df[df['file_name'].apply(lambda x: x.startswith("{}/{}".format(TUSZ_folder, pat_num)))]
    return df['full_path'].tolist()


def main():
    # pat_num = sys.argv[1]
    pat_num = -1
    filenames_list = get_filename_list(pat_num)
    # for filename in tqdm(filenames_list, desc="Files", position=0, leave=True):
    # dir_output = '../input/TUSZ_STFT/{}/{}/'.format(TUSZ_folder, pat_num)
    # random.shuffle(filenames_list)
    for filename in tqdm(filenames_list, desc="STFT extracting"):
        # file_prepared = [f.split('_band_zc.pickle')[0] for f in listdir(dir_output) if isfile(join(dir_output, f))]
        # if filename.split('/')[-1] in file_prepared:
        #     print("{} is already prepared.".format(filename))
        #     continue
        f = pyedflib.EdfReader(join("../input/TUSZ/edf/", filename))
        signal_labels = f.getSignalLabels()
        indices = get_EEG_index(signal_labels)
        signal_shape = f.readSignal(0).shape
        sample_frequency = f.getSampleFrequencies()
        duration = f.getFileDuration()
        # filename_fs_dict[filename] = {'fs': sample_frequency, 'ch': signal_labels}
        fs = int(sample_frequency[0])

        signals = np.zeros((len(EEG_channels), signal_shape[0]))
        for row, EEG_index in enumerate(indices):
            if EEG_index == -1:
                continue
            signals[row, :] = f.readSignal(EEG_index)

        sos = signal.butter(4, [0.2, 50], 'bandpass', fs=fs, output='sos')
        allsigFilt = signal.sosfiltfilt(sos, signals, axis=1)
        # all_features = get_Eglass_features(allsigFilt, fs)
        all_features = get_stft_features(allsigFilt, fs)

        # with open('../input/TUSZ_STFT/{}/{}/{}_STFT.pickle'.format(TUSZ_folder, pat_num, filename.split('/')[-1]), 'wb') as zc_file:
        #     pickle.dump(all_features, zc_file)

        with open('../input/TUSZ_STFT/{}_STFT.pickle'.format(filename.split('/')[-1]), 'wb') as zc_file:
            pickle.dump(all_features, zc_file)


def visualize():
    with open('../input/TUSZ_zc/train/02_tcp_le/000/00000002_s004_t000.edf_band_mean_ll.pickle', 'rb') as zc_file:
        data = pickle.load(zc_file)
        print(data.shape)
        plt.imshow(data)
        plt.show()


if __name__ == '__main__':
    main()
    # visualize()
