pat_file_list = [
    'Patient_5_75.mat', 'Patient_1_154.mat', 'Patient_13_109.mat', 'Patient_21_146.mat', 'Patient_30_44.mat',
    'Patient_1_55.mat', 'Patient_14_105.mat', 'Patient_15_110.mat', 'Patient_6_26.mat', 'Patient_12_66.mat',
    'Patient_27_160.mat', 'Patient_22_81.mat', 'Patient_8_95.mat', 'Patient_6_94.mat', 'Patient_7_107.mat',
    'Patient_30_27.mat', 'Patient_21_152.mat', 'Patient_1_74.mat', 'Patient_26_40.mat', 'Patient_25_130.mat',
    'Patient_9_64.mat', 'Patient_6_20.mat', 'Patient_13_141.mat', 'Patient_4_100.mat', 'Patient_13_83.mat',
    'Patient_3_71.mat', 'Patient_21_157.mat', 'Patient_3_80.mat', 'Patient_25_118.mat', 'Patient_3_84.mat',
    'Patient_2_66.mat', 'Patient_13_133.mat', 'Patient_17_98.mat', 'Patient_4_118.mat', 'Patient_6_77.mat',
    'Patient_26_37.mat', 'Patient_25_31.mat', 'Patient_20_124.mat', 'Patient_26_85.mat', 'Patient_8_66.mat',
    'Patient_21_85.mat', 'Patient_1_165.mat', 'Patient_19_77.mat', 'Patient_3_142.mat', 'Patient_26_51.mat',
    'Patient_4_121.mat', 'Patient_29_146.mat', 'Patient_24_58.mat', 'Patient_12_26.mat', 'Patient_4_105.mat',
    'Patient_15_118.mat', 'Patient_19_3.mat', 'Patient_16_143.mat', 'Patient_21_87.mat', 'Patient_9_48.mat',
    'Patient_21_149.mat', 'Patient_2_103.mat', 'Patient_24_84.mat', 'Patient_20_22.mat', 'Patient_23_129.mat',
    'Patient_2_48.mat', 'Patient_21_101.mat', 'Patient_18_65.mat', 'Patient_23_216.mat', 'Patient_19_53.mat',
    'Patient_2_98.mat', 'Patient_3_68.mat', 'Patient_5_92.mat', 'Patient_10_53.mat', 'Patient_10_89.mat',
    'Patient_16_146.mat', 'Patient_24_149.mat', 'Patient_7_60.mat', 'Patient_1_47.mat', 'Patient_14_159.mat',
    'Patient_18_60.mat', 'Patient_14_93.mat', 'Patient_28_35.mat', 'Patient_30_46.mat', 'Patient_1_26.mat',
    'Patient_13_9.mat', 'Patient_19_45.mat', 'Patient_22_52.mat', 'Patient_5_81.mat', 'Patient_6_72.mat',
    'Patient_8_24.mat', 'Patient_28_115.mat', 'Patient_12_65.mat', 'Patient_5_52.mat', 'Patient_5_68.mat',
    'Patient_26_87.mat', 'Patient_1_140.mat', 'Patient_14_112.mat', 'Patient_10_87.mat', 'Patient_6_76.mat',
    'Patient_23_177.mat', 'Patient_27_165.mat', 'Patient_21_116.mat', 'Patient_13_112.mat', 'Patient_29_205.mat',
    'Patient_12_95.mat', 'Patient_20_59.mat', 'Patient_10_36.mat', 'Patient_29_222.mat', 'Patient_15_79.mat',
    'Patient_21_159.mat', 'Patient_23_269.mat', 'Patient_29_209.mat', 'Patient_12_70.mat', 'Patient_21_132.mat',
    'Patient_27_141.mat', 'Patient_21_136.mat', 'Patient_9_51.mat', 'Patient_19_68.mat', 'Patient_7_140.mat',
    'Patient_16_126.mat', 'Patient_18_63.mat', 'Patient_13_41.mat', 'Patient_25_83.mat', 'Patient_18_51.mat',
    'Patient_11_75.mat', 'Patient_15_142.mat', 'Patient_24_143.mat', 'Patient_16_97.mat', 'Patient_1_148.mat',
    'Patient_25_128.mat', 'Patient_26_63.mat', 'Patient_4_125.mat', 'Patient_16_71.mat', 'Patient_11_105.mat',
    'Patient_9_76.mat', 'Patient_7_121.mat', 'Patient_3_78.mat', 'Patient_26_91.mat', 'Patient_8_131.mat',
    'Patient_26_89.mat', 'Patient_15_119.mat', 'Patient_10_117.mat', 'Patient_5_101.mat', 'Patient_12_24.mat',
    'Patient_15_109.mat', 'Patient_10_114.mat', 'Patient_4_103.mat', 'Patient_13_66.mat', 'Patient_28_138.mat',
    'Patient_13_135.mat', 'Patient_5_72.mat', 'Patient_25_78.mat', 'Patient_30_83.mat', 'Patient_21_141.mat',
    'Patient_1_124.mat', 'Patient_27_134.mat', 'Patient_29_91.mat', 'Patient_8_116.mat', 'Patient_28_141.mat',
    'Patient_18_39.mat', 'Patient_29_237.mat', 'Patient_28_39.mat', 'Patient_26_33.mat', 'Patient_15_129.mat',
    'Patient_12_42.mat', 'Patient_10_41.mat', 'Patient_17_77.mat', 'Patient_5_95.mat', 'Patient_20_149.mat',
    'Patient_3_91.mat', 'Patient_29_214.mat', 'Patient_20_63.mat', 'Patient_6_22.mat', 'Patient_10_88.mat',
    'Patient_6_31.mat', 'Patient_21_43.mat', 'Patient_25_103.mat', 'Patient_29_231.mat', 'Patient_15_94.mat',
    'Patient_21_133.mat', 'Patient_21_121.mat', 'Patient_28_52.mat', 'Patient_2_79.mat', 'Patient_6_45.mat',
    'Patient_7_132.mat', 'Patient_10_109.mat', 'Patient_21_139.mat', 'Patient_26_75.mat', 'Patient_7_84.mat',
    'Patient_25_108.mat', 'Patient_23_98.mat', 'Patient_8_137.mat', 'Patient_30_101.mat', 'Patient_28_18.mat',
    'Patient_22_156.mat', 'Patient_7_146.mat', 'Patient_8_118.mat', 'Patient_30_58.mat', 'Patient_24_105.mat',
    'Patient_2_59.mat', 'Patient_27_111.mat', 'Patient_14_63.mat', 'Patient_13_97.mat', 'Patient_6_1.mat',
    'Patient_11_34.mat', 'Patient_21_114.mat', 'Patient_21_153.mat', 'Patient_22_23.mat', 'Patient_22_24.mat',
    'Patient_3_94.mat', 'Patient_1_37.mat', 'Patient_21_143.mat', 'Patient_3_96.mat', 'Patient_9_68.mat',
    'Patient_24_156.mat', 'Patient_6_53.mat', 'Patient_10_37.mat', 'Patient_14_87.mat', 'Patient_6_42.mat',
    'Patient_17_69.mat', 'Patient_21_158.mat', 'Patient_17_75.mat', 'Patient_19_94.mat', 'Patient_11_24.mat',
    'Patient_11_111.mat', 'Patient_8_117.mat', 'Patient_17_37.mat', 'Patient_8_81.mat', 'Patient_13_131.mat',
    'Patient_2_81.mat', 'Patient_22_21.mat', 'Patient_30_52.mat', 'Patient_3_99.mat', 'Patient_1_113.mat',
    'Patient_24_141.mat', 'Patient_10_108.mat', 'Patient_25_56.mat', 'Patient_14_65.mat', 'Patient_22_71.mat',
    'Patient_16_121.mat', 'Patient_23_268.mat', 'Patient_23_111.mat', 'Patient_7_130.mat', 'Patient_4_91.mat',
    'Patient_24_162.mat', 'Patient_19_124.mat', 'Patient_13_130.mat', 'Patient_13_128.mat', 'Patient_9_87.mat',
    'Patient_14_133.mat', 'Patient_29_212.mat', 'Patient_30_137.mat', 'Patient_21_126.mat', 'Patient_20_145.mat',
    'Patient_14_138.mat', 'Patient_10_116.mat', 'Patient_19_16.mat', 'Patient_23_195.mat', 'Patient_4_84.mat',
    'Patient_24_127.mat', 'Patient_2_55.mat', 'Patient_8_106.mat', 'Patient_13_88.mat', 'Patient_19_100.mat',
    'Patient_18_24.mat']

EEG_channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

EEG_channels_LE = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                'EEG T5-LE', 'EEG T6-LE', 'EEG A1-LE', 'EEG A2-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE']

# feature_size = 126 if dataset == "TUSZ" else 144
# train_len = 3050138 if dataset == "TUSZ" else 15060645
# val_len = 2455 if dataset == "TUSZ" else 7192
# test_len = 552554

dataset_parameter = {"TUSZ": {
    "pretrain": {
        "feature_size": 126,
        "train_len": 3050138,
        "val_len": 2455,
        "test_len": 0
    },
    "scratch": {
        "feature_size": 126,
        "train_len": 2647984,
        "val_len": 44103,
        "test_len": 609180
    }
},
    "Epilepsiae": {
        "pretrain": {
            "feature_size": 126,
            "train_len": 3050138,  # TODO: update the length
            "val_len": 2455,
            "test_len": 0
        },
        "scratch": {
            "feature_size": 126,
            "train_len": 3050138,  # TODO: update the length
            "val_len": 2455,
            "test_len": 0
        }

    }
}

