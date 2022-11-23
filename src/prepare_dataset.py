import matplotlib.pyplot as plt
import numpy as np
import scipy.io, scipy.integrate
import os
import pickle
# from eglass import calculateMLfeatures
from utils.params import pat_file_list, EEG_channels, EEG_channels_LE
import pyedflib

# fs = 250
TUSZ_folder = "dev/02_tcp_le"


def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0


def bandpower(f, Pxx, fmin, fmax):
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return scipy.integrate.trapz(Pxx[ind_min: ind_max + 1], f[ind_min: ind_max + 1])


def calculateOtherMLfeatures_oneCh(X, fs):
    numFeat = 6 #54 from Sopic2018 and LL and meanAmpl
    lenSig= len(X)
    segLenIndx = int(4 * fs)  # length of EEG segments in samples
    slidWindStepIndx = int(1 * fs)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    featureValues=np.zeros((len(index), numFeat))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]
        f, Pxx = scipy.signal.periodogram(sig, fs=fs)
        p_delta = bandpower(f, Pxx, 0.5, 4)
        p_theta = bandpower(f, Pxx, 4, 8)
        p_alfa = bandpower(f, Pxx, 8, 13)
        p_beta = bandpower(f, Pxx, 13, 30)
        meanAmpl = np.mean(np.abs(sig))
        LL = np.mean(np.abs(np.diff(sig)))
        featureValues[i, :] = np.hstack((meanAmpl, LL, p_delta, p_theta, p_alfa, p_beta))
    return featureValues


def calculateMovingAvrgMeanWithUndersampling_v2(data, winLen, winStep):
    lenSig=len(data)
    index = np.arange(0, lenSig - winLen, winStep)

    segmData = np.zeros(len(index))
    for i in range(len(index)): #-1
        x = data[index[i]:index[i] + winLen]
        segmData[i]=np.mean(x)
    return(segmData)

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
        frg = arr[first:last+1]
        leng = last-first+1
        dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1], [frg[0], frg[-1]]))
        idx = np.argmax(dist)
        return (dist[idx], first+idx)

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
            stack.extend([(first, idx),(idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))


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
#%%


from scipy import signal
from os import listdir
from os.path import isfile, join
import sys


def main():
    print(sys.argv)
    pat_num = sys.argv[1]
    filenames_list = []
    # dir_total_files = '../TUSZ/edf/{}/{}'.format(TUSZ_folder, pat_num)
    # for dev_train in ["dev", "train"]:
    #     for dir_tcp in ["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"]:
    #         TUSZ_folder = "{}/{}".format(dev_train, dir_tcp)
            # dir_total_pats = '../TUSZ/edf/{}'.format(TUSZ_folder)
            # for pat_num in listdir(dir_total_pats):
    dir_total_files = '../TUSZ/edf/{}/{}'.format(TUSZ_folder, pat_num)
    for subdir in listdir(dir_total_files):
        for subsubdir in listdir(join(dir_total_files, subdir)):
            filenames_list += [join(dir_total_files, subdir, subsubdir, edf_file) for edf_file
                               in listdir(join(dir_total_files, subdir, subsubdir)) if edf_file.endswith('.edf')]
    print(filenames_list)

    # dir_output = '../TUSZ_zc/{}/{}'.format(TUSZ_folder, pat_num)

    # file_prepared = [f.split('_zc.pickle')[0] for f in listdir(dir_output) if isfile(join(dir_output, f))]
    # print(file_prepared)

    filename_fs_dict = {}

    for filename in filenames_list:
        # if filename in pat_file_list:
        #     continue
        # if filename.split('/')[-1] in file_prepared:
        #     print("{} is already prepared.".format(filename))
        #     continue
        print(filename)
        # data = scipy.io.loadmat('{}/{}'.format(dir_total_files, filename))
        # signals = data['Signals']
        f = pyedflib.EdfReader(filename)
        signal_labels = f.getSignalLabels()
        indices = get_EEG_index(signal_labels)
        signal_shape = f.readSignal(0).shape
        sample_frequency = f.getSampleFrequencies()
        filename_fs_dict[filename] = {'fs': sample_frequency, 'ch': signal_labels}
        print('Frequency: {}'.format(sample_frequency))
        fs = int(sample_frequency[0])

        signals = np.zeros((len(EEG_channels), signal_shape[0]))
        for row, EEG_index in enumerate(indices):
            if EEG_index == -1:
                continue
            signals[row, :] = f.readSignal(EEG_index)

        sos = signal.butter(4, [1, 20], 'bandpass', fs=fs, output='sos')
        allsigFilt = signal.sosfiltfilt(sos, signals, axis=1)

        #%%
        numCh = len(EEG_channels)
        num_feat = 6
        win_len = 4
        EPS_thresh_arr=[16, 32, 64, 128, 256]
        length = int(allsigFilt.shape[1] // fs) - win_len


        # zeroCrossStandard = np.zeros((length, numCh))
        # zeroCrossApprox = np.zeros((length, numCh))
        # zeroCrossFeaturesAll = np.zeros((length, num_feat * numCh))

        for ch in range(numCh):
            sigFilt=allsigFilt[ch, :(allsigFilt.shape[1] // fs)*fs]

            featOther = calculateOtherMLfeatures_oneCh(np.copy(sigFilt), fs)
            if (ch == 0):
                AllFeatures = featOther
            else:
                AllFeatures = np.hstack((AllFeatures, featOther))

            # x = np.convolve(zero_crossings(sigFilt), np.ones(fs), mode='same')
            # zeroCrossStandard = calculateMovingAvrgMeanWithUndersampling_v2(x, fs * 4, fs)
            # zeroCrossFeaturesAll[:, num_feat * ch] = zeroCrossStandard
            # for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
            #     sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)
            #     sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
            #     x = np.convolve(zero_crossings(sigApproxInterp), np.ones(fs), mode='same')
            #     zeroCrossApprox = calculateMovingAvrgMeanWithUndersampling_v2(x, fs *4 , fs)
            #     zeroCrossFeaturesAll[:, num_feat * ch + EPSthrIndx + 1] = zeroCrossApprox
        with open('../input/TUSZ_zc/{}/{}/{}_band_mean_ll.pickle'.format(TUSZ_folder, pat_num, filename.split('/')[-1]), 'wb') as zc_file:
            pickle.dump(AllFeatures, zc_file)
    # print(filename_fs_dict)
    # with open('../TUSZ_zc/fs.pickle', 'wb') as zc_file:
    #     pickle.dump(filename_fs_dict, zc_file)


def visualize():
    with open('../input/TUSZ_zc/train/02_tcp_le/000/00000002_s004_t000.edf_band_mean_ll.pickle', 'rb') as zc_file:
        data = pickle.load(zc_file)
        print(data.shape)
        plt.imshow(data)
        plt.show()


if __name__ == '__main__':
    main()
    # visualize()
