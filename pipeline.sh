#!/bin/bash

while true; do
    # Check if the file exists
    if [ ! -f "ended.txt" ]; then
        echo "File does not exist."
        
        # Check if a Python program is running
        if ! pgrep -x "python" > /dev/null; then
            echo "Python is not running."
            echo "Conditions not met. Launching python generate.py..."
            python generate.py &
        fi
    else
        echo "File exists. Exiting loop."
        break  # Exit the while loop
    fi

    # Wait for 60 seconds before checking again
    sleep 60
done

python scoring.py

python plot.py

