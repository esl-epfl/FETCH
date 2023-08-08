#!/bin/bash

# Loop from 0 to 10 (inclusive)
cd code
for channel_id in {7..10}; do
    echo "Running inference.py with selected_channel_id=$channel_id"
    python inference.py --selected_channel_id "$channel_id" 
done

