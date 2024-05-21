#!/bin/bash

# edit_input
#
# Finds directories including INPUT_FILE_NAME.
# Navigates to paths (using pushd/popd) and modifies input files.
#
# Author: Gokhan Oztarhan
# Created date: 04/03/2023
# Last modified: 04/03/2023

INPUT_FILE_NAME="input.json"
ROOT_DIR="."

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
    
    sed -i '2s/.*/"verbose_file": 0,/' $INPUT_FILE_NAME
    sed -i '5s/.*/"root_dir": ".",/' $INPUT_FILE_NAME
    sed -i '27s/.*/"delta_E_lim": 1e-15,/' $INPUT_FILE_NAME
    
    popd > /dev/null
done
