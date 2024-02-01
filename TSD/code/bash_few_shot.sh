#!/bin/bash

#python3 few_shot_train.py --save_directory /scrap/users/amirshah/TUH_STFT_data/ --num_nodes 8 --experiment_root ~/TUH_STFT_data/output/model_8nodes/ --learning_rate 3e-5 --lr_scheduler_step 200
# I need a loop to run the above command for variable instead of 8 nodes I need 2 to 20 nodes
# The name of the path should be adapted to the number of nodes like model_2nodes, model_3nodes, etc.
for i in {2..20}
do
    echo "Running few_shot_train.py with num_nodes=$i"
    python3 few_shot_train.py --save_directory /scrap/users/amirshah/TUH_STFT_data/ --num_nodes $i --experiment_root ~/TUH_STFT_data/output/model_${i}nodes/ --learning_rate 3e-5 --lr_scheduler_step 100
done

