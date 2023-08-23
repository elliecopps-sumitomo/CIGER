#!/bin/usr/env bash

cd ../GSEA_Linux_4.3.2/

file_path="../data/gsea_expressions"

for file in "$file_path"/*; do
    if [ -f "$file" ]; then
        echo "$file"
    fi
done