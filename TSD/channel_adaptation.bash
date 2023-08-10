#!/bin/bash
# Export the directory containing parser.py if not already done
if [[ -z "${PYTHONPATH}" ]]; then
    export PARSER_DIR=$(dirname $(pwd))
    export PYTHONPATH=$PARSER_DIR
fi

cd code

# Generate random numbers between 0 and 7542 using shuf command
random_numbers=$(shuf -i 0-7542 -n 50)

# Loop through the generated random numbers
for channel_id in $random_numbers; do
    echo "Running inference.py with selected_channel_id=$channel_id"
    python few_shot_train.py --selected_channel_id "$channel_id"
    python inference.py --selected_channel_id "$channel_id" --global
    python inference.py --selected_channel_id "$channel_id" 
    # Store the exit code in a variable
    exit_code=$?
   if [ "$1" = "train" ] && [ $exit_code -eq 2 ]; then
        echo "inference.py returned not exists. Running train.py."
        python best_model.py --selected_channel_id "$channel_id"
	python inference.py --selected_channel_id "$channel_id"
   fi
done



