#!/bin/bash

# Define input parameters
start_date="20110101"
end_date="20110701"
interval_days=1
directory="/fs/scratch/PAS2856/AS4194_Project/PatelShuvo"
output_folder="variance_plots"

# Define variables and ensemble types
variables=("u" "v" "t" "q")
ensemble_types=("perturbed_ens" "reference_ens")

# Create the output folder if it doesn't exist
mkdir -p $output_folder

# Loop over variables and ensemble types
for variable in "${variables[@]}"; do
    for ensemble in "${ensemble_types[@]}"; do
        echo "Processing $variable with $ensemble..."
        output_file="${output_folder}/${variable}_${ensemble}_variance_plot.png"
        
        # Run the Python script
        python examine_ens_variance_4.py "$start_date" "$end_date" "$interval_days" "$variable" "$ensemble"
        
        # Move the output image to the new folder with an appropriate name
        mv sample_figure4_final.png "$output_file"
    done
done

echo "All variance plots have been generated and saved in the folder: $output_folder"
