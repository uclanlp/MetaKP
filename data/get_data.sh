#!/bin/bash

HOME_DIR=`realpath ../`

# Extract all .tar.gz files to get data
find "$HOME_DIR" -type f -name '*.tar.gz' | while read -r file; do
    tar -xzf "$file" -C "$(dirname "$file")"
    echo "Extracted $(basename "$file") to $(dirname "$file")"
done

