# FETCH: A Fast and Efficient Technique for Channel Selection in EEG Wearable Systems 

## Overview
This repository contains the source code and documentation for our recent study on the channel adaptation methodology in electroencephalography (EEG) signals using few-shot learning in wearable biomedical systems. 

**Paper Link**: [Read the Manuscript](https://infoscience.epfl.ch/record/310332?ln=en&v=pdf)

**Lab Website**: [Embedded Systems Laboratory (ESL) at EPFL](https://www.epfl.ch/labs/esl/)

## Simplified Abstract
In our study, we tackle the challenge of efficiently selecting the optimal set of electrodes for EEG-based seizure detection. Traditional methods require extensive computational resources, exploring limited electrode combinations. We propose a novel approach that dramatically speeds up this process by using a method inspired by few-shot learning, allowing us to evaluate all possible combinations without the need to retrain the network extensively. Our technique reduces what would traditionally take months into just a few hours on a single GPU, achieving high accuracy with significantly less effort.

## Main Dataset
The research utilizes the [TUH EEG Seizure Corpus (TUSZ)](https://isip.piconepress.com/projects/tuh_eeg/), an open-source dataset for the seizure detection task.

## Contributions
- **Efficient Exploration of Electrode Configurations**: By leveraging few-shot learning, our method can explore all potential electrode combinations quickly and effectively.
- **High Performance with Reduced Training Time**: Achieves performance matching with previous research results but in much less time.
- **Practical Application in Epileptic Seizure Detection**: Demonstrates the feasibility of our approach in real-world settings, potentially transforming the development and usability of wearable EEG systems.

## Contact
For more information or to discuss this research, feel free to contact Alireza Amirshahi at [Alireza's EPFL profile](https://people.epfl.ch/alireza.amirshahi/?lang=en) or Professor David Atienza at [Prof. David Atienza's EPFL profile](https://people.epfl.ch/david.atienza?lang=en).

