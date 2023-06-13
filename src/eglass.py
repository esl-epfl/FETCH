import numpy as np
import scipy.signal, scipy.integrate
from sklearn.ensemble import RandomForestClassifier
import pywt
import antropy
import pickle


def sh_ren_ts_entropy(x, a, q):
    p, bin_edges = np.histogram(np.array(x, dtype=np.float32))
    p = p / np.sum(p)
    p = p[np.where(p > 0)]  # to exclude log(0)
    shannon_en = - np.sum(p * np.log2(p))
    renyi_en = np.log2(np.sum(pow(p, a))) / (1 - a)
    tsallis_en = (1 - np.sum(pow(p, q))) / (q - 1)
    return (shannon_en, renyi_en, tsallis_en)


def bandpower(f, Pxx, fmin, fmax):
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return scipy.integrate.trapz(Pxx[ind_min: ind_max + 1], f[ind_min: ind_max + 1])


def sampen(m, r, L):
    epsilon = 0.001
    N = len(L)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([L[i: i + m] for i in range(N - m)])
    xmj = np.array([L[i: i + m] for i in range(N - m + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([L[i: i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log((A + epsilon) / (B + epsilon))


def sampen2(dim, r, data):
    epsilon = 0.001
    N = len(data)
    correl = np.zeros(2)
    dataMat = np.zeros((dim + 1, N - dim))
    for i in range(dim + 1):

        dataMat[i, :] = data[i: N - dim + i]

    for m in range(dim, dim + 2):
        count = np.zeros(N - dim)
        tempMat = dataMat[0:m, :]

        for i in range(N - m):
            # calculate distance, excluding self - matching case
            dist = np.max(np.abs(tempMat[:, i + 1: N - dim] - np.tile(tempMat[:, i], ((N - dim - i - 1), 1)).T), axis=0)
            D = (dist < r)
            count[i] = np.sum(D) / (N - dim - 1)

        correl[m - dim] = np.sum(count) / (N - dim)

    saen = np.log((correl[0] + epsilon) / (correl[1] + epsilon))
    return saen


def calculateMLfeatures(data, samplFreq):
    ''' function that calculates various features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
    at the bottom of function choose whether 45 or 54 features are used
    data is a 1D array representing data window from which to extract features
    '''
    # some parameters
    DWTfilterName = 'db4'  # 'sym5'
    DWTlevel = 7
    n1 = 2  # num dimensions for sample entropy
    r1 = 0.2  # num of STD for sample entropy
    r2 = 0.35  # num of STD for sample entropy
    a = 2  # param for shannon, renyi and tsallis enropy
    q = 2  # param for shannon, renyi and tsallis enropy

    # DWT
    coeffs = pywt.wavedec(data, DWTfilterName, level=DWTlevel)
    a7, d7, d6, d5, d4, d3, d2, d1 = coeffs

    # sample entropy
    samp_1_d7_1 = sampen2(n1, r1 * np.std(d7), d7)
    samp_1_d6_1 = sampen2(n1, r1 * np.std(d6), d6)
    samp_2_d7_1 = sampen2(n1, r2 * np.std(d7), d7)
    samp_2_d6_1 = sampen2(n1, r2 * np.std(d6), d6)

    # permutation entropy
    # perm_d7_3 = perm_entropy(d7, order=3, delay=1, normalize=False)
    # perm_d7_5 = perm_entropy(d7, order=5, delay=1, normalize=False)
    # perm_d7_7 = perm_entropy(d7, order=7, delay=1, normalize=False)
    # perm_d6_3 = perm_entropy(d6, order=3, delay=1, normalize=False)
    # perm_d6_5 = perm_entropy(d6, order=5, delay=1, normalize=False)
    # perm_d6_7 = perm_entropy(d6, order=7, delay=1, normalize=False)
    # perm_d5_3 = perm_entropy(d5, order=3, delay=1, normalize=False)
    # perm_d5_5 = perm_entropy(d5, order=5, delay=1, normalize=False)
    # perm_d5_7 = perm_entropy(d5, order=7, delay=1, normalize=False)
    # perm_d4_3 = perm_entropy(d4, order=3, delay=1, normalize=False)
    # perm_d4_5 = perm_entropy(d4, order=5, delay=1, normalize=False)
    # perm_d4_7 = perm_entropy(d4, order=7, delay=1, normalize=False)
    # perm_d3_3 = perm_entropy(d3, order=3, delay=1, normalize=False)
    # perm_d3_5 = perm_entropy(d3, order=5, delay=1, normalize=False)
    # perm_d3_7 = perm_entropy(d3, order=7, delay=1, normalize=False)

    perm_d7_3 = antropy.perm_entropy(d7, order=3, delay=1, normalize=True)
    perm_d7_5 = antropy.perm_entropy(d7, order=5, delay=1, normalize=True)
    perm_d7_7 = antropy.perm_entropy(d7, order=7, delay=1, normalize=True)
    perm_d6_3 = antropy.perm_entropy(d6, order=3, delay=1, normalize=True)
    perm_d6_5 = antropy.perm_entropy(d6, order=5, delay=1, normalize=True)
    perm_d6_7 = antropy.perm_entropy(d6, order=7, delay=1, normalize=True)
    perm_d5_3 = antropy.perm_entropy(d5, order=3, delay=1, normalize=True)
    perm_d5_5 = antropy.perm_entropy(d5, order=5, delay=1, normalize=True)
    perm_d5_7 = antropy.perm_entropy(d5, order=7, delay=1, normalize=True)
    perm_d4_3 = antropy.perm_entropy(d4, order=3, delay=1, normalize=True)
    perm_d4_5 = antropy.perm_entropy(d4, order=5, delay=1, normalize=True)
    perm_d4_7 = antropy.perm_entropy(d4, order=7, delay=1, normalize=True)
    perm_d3_3 = antropy.perm_entropy(d3, order=3, delay=1, normalize=True)
    perm_d3_5 = antropy.perm_entropy(d3, order=5, delay=1, normalize=True)
    perm_d3_7 = antropy.perm_entropy(d3, order=7, delay=1, normalize=True)

    # shannon renyi and tsallis entropy
    (shannon_en_sig, renyi_en_sig, tsallis_en_sig) = sh_ren_ts_entropy(data, a, q)
    (shannon_en_d7, renyi_en_d7, tsallis_en_d7) = sh_ren_ts_entropy(d7, a, q)
    (shannon_en_d6, renyi_en_d6, tsallis_en_d6) = sh_ren_ts_entropy(d6, a, q)
    (shannon_en_d5, renyi_en_d5, tsallis_en_d5) = sh_ren_ts_entropy(d5, a, q)
    (shannon_en_d4, renyi_en_d4, tsallis_en_d4) = sh_ren_ts_entropy(d4, a, q)
    (shannon_en_d3, renyi_en_d3, tsallis_en_d3) = sh_ren_ts_entropy(d3, a, q)

    # band power
    f, Pxx = scipy.signal.periodogram(data, fs=samplFreq)
    p_tot = bandpower(f, Pxx, 0, 45)
    p_dc = bandpower(f, Pxx, 0, 0.5)
    p_mov = bandpower(f, Pxx, 0.1, 0.5)
    p_delta = bandpower(f, Pxx, 0.5, 4)
    p_theta = bandpower(f, Pxx, 4, 8)
    p_alfa = bandpower(f, Pxx, 8, 13)
    p_middle = bandpower(f, Pxx, 12, 13)
    p_beta = bandpower(f, Pxx, 13, 30)
    p_gamma = bandpower(f, Pxx, 30, 45)
    p_dc_rel = p_dc / p_tot if p_tot!=0 else 0
    p_mov_rel = p_mov / p_tot if p_tot!=0 else 0
    p_delta_rel = p_delta / p_tot if p_tot!=0 else 0
    p_theta_rel = p_theta / p_tot if p_tot!=0 else 0
    p_alfa_rel = p_alfa / p_tot if p_tot!=0 else 0
    p_middle_rel = p_middle / p_tot if p_tot!=0 else 0
    p_beta_rel = p_beta / p_tot if p_tot!=0 else 0
    p_gamma_real = p_gamma / p_tot if p_tot!=0 else 0

    # all features from the paper - 54 features
    featuresAll= [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3, perm_d6_5, perm_d6_7,   perm_d5_3, perm_d5_5,
             perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig, renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7,
             shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4, renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3,
             p_tot, p_dc, p_mov, p_delta, p_theta, p_alfa, p_middle, p_beta, p_gamma, p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_real]
    # here I exclude features that are not normalized - 45 features
    # featuresAll = [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3,
    #                perm_d6_5, perm_d6_7, perm_d5_3, perm_d5_5,
    #                perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig,
    #                renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7,
    #                shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4,
    #                renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3,
    #                p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_real]
    return (featuresAll)


if __name__ == '__main__':
    signal_fake = np.random.random(30000)
    features1 = calculateMLfeatures(signal_fake[:10000], 200)
    features2 = calculateMLfeatures(signal_fake[10000:20000], 200)
    features3 = calculateMLfeatures(signal_fake[20000:], 200)
    features_total = calculateMLfeatures(signal_fake, 200)
    features_mean = np.mean((features1, features2, features3), axis=0)
    print(features1)
    print(features2)
    print(features3)
    # print(features_mean)
    print(features_total)
