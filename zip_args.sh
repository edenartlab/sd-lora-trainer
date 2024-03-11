#!/bin/bash

# Check if the target directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 target_directory"
  exit 1
fi

# Check if the target directory exists
if [ ! -d "$1" ]; then
  echo "Error: directory '$1' doesn't exist."
  exit 1
fi

# Navigate to the target directory
cd "$1" || exit 1

# Initialize empty zip file
zip -r9 "args.zip" --exclude=*

# Find and zip all training_args.json files
find . -type f -name 'training_args.json' -exec zip -r "args.zip" {} +

# Navigate back to the original directory
cd - || exit 1

echo "All training_args.json files have been zipped into args.zip"  