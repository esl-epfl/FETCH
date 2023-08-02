import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
import pickle
from src.prepare_dataset import zero_crossings, calculateMovingAvrgMeanWithUndersampling_v2
from src.prepare_dataset import calculateOtherMLfeatures_oneCh, polygonal_approx

from multiprocessing import Pool
from functools import partial

from sklearn.ensemble import RandomForestClassifier
import joblib


def get_features(signals):
    """
    Extracts features from the input signals.

    Input:
    signals (numpy.ndarray): Input signals with shape (numCh, signal_len), where numCh is the number of channels
                             and signal_len is the length of each signal.

    Returns:
    numpy.ndarray: Array containing the extracted features for each channel.

    Description:
    This function takes an input array of signals and extracts various features for each channel.
    The features include zero-crossing, band powers, LL (Low-Level) features, and a scale factor.
    The function first scales the input signals to a range of [-1, 1] and then calculates the features.
    """

    FS = 256 # Define the sampling frequency
    # %%
    numCh = signals.shape[0]  # Number of channels
    num_feat = 7  # Number of features to be extracted for each channel
    time_len = 12  # Length of each signal in seconds for calculating moving average features
    eps = 1e-6  # A small epsilon value to avoid division by zero

    # Threshold values for polygonal approximation
    EPS_thresh_arr = [0.01, 0.04, 0.1, 0.4, 0.8]

    all_features = np.zeros((num_feat * numCh))  # Zero-crossing, band powers, LL, and scale_factor

    # Calculate the scale factor for each channel to normalize the signals
    scaleFactor = np.max(signals, axis=1, keepdims=True) - np.min(signals, axis=1, keepdims=True)
    signalsScaled = (signals - np.min(signals, axis=1, keepdims=True)) / (scaleFactor + eps)
    signalsScaled = signalsScaled * 2 - 1

    for ch in range(numCh):
        signal_ch = signalsScaled[ch, :]

        # Calculate other Machine Learning features for the channel
        featOther = calculateOtherMLfeatures_oneCh(np.copy(signal_ch), FS, time_len, time_len)

        # Store the calculated features in the all_features array
        all_features[ch * num_feat:ch * num_feat + 5] = featOther
        all_features[ch * num_feat+ 5] = scaleFactor[ch] / 5000

        # Calculate zero-crossing feature for the original signal
        x = np.convolve(zero_crossings(signal_ch), np.ones(FS), mode='same')

        zeroCrossStandard = calculateMovingAvrgMeanWithUndersampling_v2(x, FS * time_len, FS)

        all_features[ch * num_feat + 6] = zeroCrossStandard

        # Calculate zero-crossing features for polygonal approximations with different thresholds
        # for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
        #     sigApprox = polygonal_approx(signal_ch, epsilon=EPSthr)
        #     sigApproxInterp = np.interp(np.arange(len(signal_ch)), sigApprox,
        #                                 signal_ch[sigApprox])
        #     x = np.convolve(zero_crossings(sigApproxInterp), np.ones(FS), mode='same')
        #
        #     zeroCrossApprox = calculateMovingAvrgMeanWithUndersampling_v2(x, FS * time_len, FS)
        #
        #     all_features[ch * num_feat + 7 + EPSthrIndx] = zeroCrossApprox

    return all_features


def process_file(file_path, features_dir):
    try:
        # Load data from the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Call the get_features function with the loaded data
        features = get_features(data['signals'])

        # Create a new filename for the features pickle file
        features_filename = os.path.splitext(os.path.basename(file_path))[0] + "_features.pkl"
        features_file_path = os.path.join(features_dir, features_filename)

        # Save the features in a new pickle file
        with open(features_file_path, 'wb') as features_file:
            pickle.dump(features, features_file)

        # print(f"Features extracted from '{os.path.basename(file_path)}' and saved to '{features_file_path}'")

    except Exception as e:
        print(f"Error processing file '{os.path.basename(file_path)}': {str(e)}")


def process_files():
    dir_name = "../../TUSZv2/preprocess/task-binary_datatype-train"
    # Check if the directory exists
    if not os.path.exists(dir_name):
        print(f"Directory '{dir_name}' does not exist.")
        return

    # Get a list of files in the directory
    file_list = os.listdir(dir_name)

    # Create a new directory for saving the features
    features_dir = dir_name + "_features"
    os.makedirs(features_dir, exist_ok=True)

    # Use multiprocessing to process files in parallel
    num_processes = 6  # Number of available CPU cores
    with Pool(num_processes) as pool:
        # Map the file processing function to the list of files
        partial_process_file = partial(process_file, features_dir=features_dir)
        list(tqdm(pool.imap(partial_process_file, [os.path.join(dir_name, filename) for filename in file_list]),
                  total=len(file_list), desc="Processing Files", unit="file"))


def load_features_from_folder(folder_path):
    features = []
    labels = []
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith("_features.pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                feature_vector = pickle.load(file)
                features.append(feature_vector)

            # Extract the label from the filename (positive if "seiz" in filename, otherwise negative)
            if "seiz" in filename:
                label = 1
            else:
                label = 0
            labels.append(label)

    return np.array(features), np.array(labels)


def train_random_forest():
    train_folder = "../../TUSZv2/preprocess/task-binary_datatype-train_features"
    train_features, y_train = load_features_from_folder(train_folder)
    print("Train data extracted!")
    dev_folder = "../../TUSZv2/preprocess/task-binary_datatype-dev_features"
    dev_features, y_dev = load_features_from_folder(dev_folder)
    print("Dev data extracted!")
    test_folder = "../../TUSZv2/preprocess/task-binary_datatype-eval_features"
    test_features, y_test = load_features_from_folder(test_folder)
    print("Eval data extracted!")

    # Create a Random Forest Classifier with desired parameters
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)

    # Train the model on the training data
    rf_classifier.fit(train_features, y_train)

    # Evaluate the model on the development data
    accuracy = rf_classifier.score(dev_features, y_dev)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Test the model on the test data
    test_accuracy = rf_classifier.score(test_features, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Save the trained model to a file for future inferences
    model_filename = "rf_model.pkl"
    model_path = os.path.join(".", model_filename)
    joblib.dump(rf_classifier, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    # process_files()
    train_random_forest()
