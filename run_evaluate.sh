#!/bin/bash

# Define the CSV file containing your configurations
CSV_FILE="models/model_parameters_test.csv"

# Check if the file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: $CSV_FILE not found!"
    exit 1
fi

echo "Extracting model names from $CSV_FILE..."

# Extract the first column (model_name), skip the header (NR>1), 
# and remove any quotes or carriage returns
MODEL_NAMES=$(awk -F',' 'NR>1 {print $1}' "$CSV_FILE" | tr -d '"' | tr -d '\r')

# Loop through each model name and execute the pipeline
for MODEL in $MODEL_NAMES; do
    echo "=========================================================="
    echo "Starting evaluating for: $MODEL"
    echo "=========================================================="
    
    # Execute your python pipeline
    python run_pipeline.py --evaluate --models_to_run "$MODEL"
    
    # Capture the exit code to know if it crashed, but let the loop continue
    if [ $? -eq 0 ]; then
        echo "Successfully completed: $MODEL"
    else
        echo "ERROR: Pipeline failed for $MODEL. Moving to the next one..."
    fi
    echo ""
done

echo "All configurations have been processed."