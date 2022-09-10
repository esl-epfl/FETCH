import numpy as np
import scipy.io
import os
import pickle
from eglass import calculateMLfeatures
from utils.pat_names import pat_file_list

fs = 256


def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0

def calculateOtherMLfeatures_oneCh(X):
    numFeat = 56 #54 from Sopic2018 and LL and meanAmpl
    lenSig= len(X)
    segLenIndx = int(4 * fs)  # length of EEG segments in samples
    slidWindStepIndx = int(1 * fs)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    featureValues=np.zeros((len(index), numFeat))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]
        feat54 = calculateMLfeatures(sig, fs)
        meanAmpl = np.mean(np.abs(sig))
        LL = np.mean(np.abs(np.diff(sig)))
        featureValues[i, :] = np.hstack((meanAmpl, LL, feat54))
    return (featureValues)


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
#%%


from scipy import signal
from os import listdir
from os.path import isfile, join
import sys

print(sys.argv)
pat_num = sys.argv[1]

dir_total_files = '../raw_eeg/Patient_{}_all'.format(pat_num)
onlyfiles = [f for f in listdir(dir_total_files) if isfile(join(dir_total_files, f))]

dir_output = '../non_seizure'
file_prepared = [f.split('_zc.pickle')[0] for f in listdir(dir_output) if isfile(join(dir_output, f))]
print(file_prepared)
exit()

for filename in onlyfiles:
    if filename in pat_file_list:
        continue
    if filename in file_prepared:
        print("{} is already prepared.".format(filename))
        continue
    print(filename)
    data = scipy.io.loadmat('{}/{}'.format(dir_total_files, filename))
    signals = data['Signals']

    sos = signal.butter(4, [1, 20], 'bandpass', fs=fs, output='sos')
    allsigFilt = signal.sosfiltfilt(sos, signals, axis=1)

    #%%
    numCh = 24
    num_feat = 6
    win_len = 4
    EPS_thresh_arr=[16, 32, 64, 128, 256]
    length = (allsigFilt.shape[1] // fs) - win_len

    # zeroCrossStandard = np.zeros((length, numCh))
    # zeroCrossApprox = np.zeros((length, numCh))
    zeroCrossFeaturesAll = np.zeros((length, num_feat * numCh))

    for ch in range(numCh):
        sigFilt=allsigFilt[ch, :]

        # featOther = calculateOtherMLfeatures_oneCh(np.copy(sigFilt))
        # if (ch == 0):
        #     AllFeatures = featOther
        # else:
        #     AllFeatures = np.hstack((AllFeatures, featOther))

        x = np.convolve(zero_crossings(sigFilt), np.ones(fs), mode='same')
        zeroCrossStandard = calculateMovingAvrgMeanWithUndersampling_v2(x, fs * 4, fs)
        zeroCrossFeaturesAll[:, num_feat * ch] = zeroCrossStandard
        for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
            sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)
            sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
            x = np.convolve(zero_crossings(sigApproxInterp), np.ones(fs), mode='same')
            zeroCrossApprox = calculateMovingAvrgMeanWithUndersampling_v2(x, fs *4 , fs)
            zeroCrossFeaturesAll[:, num_feat * ch + EPSthrIndx + 1] = zeroCrossApprox
    with open('../non_seizure/{}_zc.pickle'.format(filename), 'wb') as zc_file:
        pickle.dump(zeroCrossFeaturesAll, zc_file)