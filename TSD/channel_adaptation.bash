#!/bin/bash

# Loop from 0 to 10 (inclusive)
cd code
for channel_id in {7..9}; do
    echo "Running inference.py with selected_channel_id=$channel_id"
    python inference.py --selected_channel_id "$channel_id"
   if [ $? -eq 2 ]; then
        echo "inference.py returned not exists. Running train.py."
        python best_model.py --selected_channel_id "$channel_id"
	python inference.py --selected_channel_id "$channel_id"
   fi 
done

