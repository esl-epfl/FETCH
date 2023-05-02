Prepare the environment:
1. docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
2. ./init-docker-config.sh {IMAGE ID}  mu-1.13

Before training and testing the model, you need to download TUSZ (v1.5.4) 
from the TUH official website and modify the corresponding data storage path in the code.

Experiment:
    Prepare data:
        1. python3 tuh_dataset.py
    Train the model:
        1. python3 best_model.py
    Test model:
        1. python3 inference.py
