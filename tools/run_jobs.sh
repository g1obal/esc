#!/bin/bash

# run_jobs
#
# Run esc using conda environment.
# Remove conda related lines to disable conda functionality.
#
# Finds directories including INPUT_FILE_NAME.
# Navigates to run path (using pushd/popd) and runs job.
#
# Waits for job to finish before executing another. 
# Runs single job at a time.
#
# BEFORE USE:
# Modify ESC_PATH, CONDA_PATH and CONDA_ENV_NAME variables.
#
# Author: Gokhan Oztarhan
# Created date: 03/12/2022
# Last modified: 04/12/2022

ROOT_DIR="."
INPUT_FILE_NAME="input.ini"

ESC_PATH="$HOME/path/to/esc"

CONDA_PATH="$HOME/path/to/conda"
CONDA_ENV_NAME="my_env_name"

# Initialize conda
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    . "$CONDA_PATH/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# Activate conda environment
conda deactivate
conda activate $CONDA_ENV_NAME

# For libGL error failed to load drivers, uncomment the following.
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# For XDG_SESSION_TYPE=wayland ignored error, uncomment the following.
#XDG_SESSION_TYPE=""

# Add esc path to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$ESC_PATH"

# Find directories including INPUT_FILE_NAME recursively.
# Does not work if directory names include white spaces or special characters!
paths=()
for D in `find $ROOT_DIR -type f -name $INPUT_FILE_NAME | sort -V`; do
    paths+=(${D%/*}) # run paths, removing INPUT_FILE_NAME from string
done

# Loop over paths
for i in ${!paths[@]}; do
    run_path=${paths[$i]}
    
    pushd $run_path > /dev/null
    
    echo $run_path
    
    python3 -m esc $INPUT_FILE_NAME 1> out 2> err &
    
    wait
    
    popd > /dev/null
done

# deactivate conda environment
conda deactivate


