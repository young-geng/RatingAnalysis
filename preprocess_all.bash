#! /bin/bash

# Preprocess all the csv under raw_data directory

for name in `ls raw_data | grep csv`; do
    python preprocess.py raw_data/$name data/$name
done

echo "Preprocessing completed!"
