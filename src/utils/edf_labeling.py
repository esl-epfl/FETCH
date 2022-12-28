import pyedflib
import numpy as np

labels_standard = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                   'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                   'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                   'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                   'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                   'O1..', 'Oz..', 'O2..', 'Iz..']


def convert_signal():
    f = pyedflib.EdfReader("../../input/chb-mit-bendr/s005/chb02_30.edf")
    signal_labels = f.getSignalLabels()
    num_channels = len(signal_labels)

    f_w = pyedflib.EdfWriter("../../input/chb-mit-bendr/s005/chb02_30_new.edf", num_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS)
    data_list = []
    for ch in range(num_channels):
        f_w.setDigitalMaximum(ch, f.getDigitalMaximum(ch))
        f_w.setDigitalMinimum(ch, f.getDigitalMinimum(ch))
        f_w.setPhysicalMaximum(ch, f.getPhysicalMaximum(ch))
        f_w.setPhysicalMinimum(ch, f.getPhysicalMinimum(ch))
        # f_w.setLabel(ch, f.getLabel(ch))
        f_w.setLabel(ch, labels_standard[ch])
        # f_w.setSignalHeader(ch, f.getSignalHeader(ch))
        f_w.setSamplefrequency(ch, f.getSampleFrequency(ch))
        data_list.append(f.readSignal(ch))

    onset, dur, desc = [0., 4.2, 8.3, 12.5, 16.6, 20.8, 24.9, 29.1, 33.2,
                        37.4, 41.5, 45.7, 49.8, 54., 58.1, 62.3, 66.4, 70.6,
                        74.7, 78.9, 83., 87.2, 91.3, 95.5, 99.6, 103.8, 107.9,
                        112.1, 116.2, 120.4], [4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2,
                                               4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1, 4.2, 4.1,
                                               4.2, 4.1, 4.2, 4.1], ['T0', 'T2', 'T0', 'T1', 'T0', 'T1', 'T0', 'T2',
                                                                     'T0', 'T2', 'T0',
                                                                     'T1', 'T0', 'T2', 'T0', 'T1', 'T0', 'T2', 'T0',
                                                                     'T1', 'T0', 'T1',
                                                                     'T0', 'T2', 'T0', 'T1', 'T0', 'T2', 'T0', 'T1']
    for i in range(len(onset)):
        f_w.writeAnnotation(onset[i], dur[i], desc[i])

    f_w.writeSamples(data_list)

    f.close()
    f_w.close()


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


if __name__ == '__main__':
    convert_signal()
    read_signal("../../input/chb-mit-bendr/s005/chb02_30_new.edf")
