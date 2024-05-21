#!/bin/bash

# run_jobs
#
# Finds directories including INPUT_FILE_NAME.
# Navigates to run path (using pushd/popd) and runs job.
#
# Waits for job to finish before executing another. 
# Runs single job at a time.
#
# BEFORE USE:
# Modify ESC_PATH variable.
#
# Author: Gokhan Oztarhan
# Created date: 03/12/2022
# Last modified: 21/05/2024

ROOT_DIR="."
INPUT_FILE_NAME="input.json"

ESC_PATH="$HOME/path/to/esc"

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
    
    python3 $ESC_PATH/esc_run.py -i $INPUT_FILE_NAME 1> out 2> err &
    
    wait
    
    popd > /dev/null
done


