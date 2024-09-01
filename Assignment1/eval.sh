#!/bin/bash

# Check if the required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <python_script.py> <input_np_file.npy>"
    exit 1
fi

# Get the Python script filename and NP filename from the command line arguments
python_script="$1"
np_file="$2"

# Check if the specified files exist
if [ ! -f "$python_script" ]; then
    echo "Error: Python script '$python_script' not found."
    exit 1
fi

if [ ! -f "$np_file" ]; then
    echo "Error: NP file '$np_file' not found."
    exit 1
fi

# Run the Python script with the provided input file
python3 "$python_script" "$np_file"
