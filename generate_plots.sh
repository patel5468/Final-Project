#!/bin/bash

# Define input parameters
start_date="20110101"
end_date="20110701"
interval_days=1
output_folder="variance_plots"

# Define variables and plot types
variables=("u" "v" "t" "q")
plot_types=("perturbed_ens" "reference_ens" "normalized")

# Create the output folder if it doesn't exist
mkdir -p $output_folder

# Loop over variables and plot types
for variable in "${variables[@]}"; do
    for plot_type in "${plot_types[@]}"; do
        echo "Processing $plot_type for $variable..."
        
        # Run the Python script for variance and normalized variance
        python examine_ens_variance_5.py "$start_date" "$end_date" "$interval_days" "$variable" "$plot_type"
        
        # Correct the output file name based on Python script output
        python_output_file="${variable}_${plot_type}_variance.png"
        output_file="${output_folder}/${variable}_${plot_type}_plot.png"

        # Move the output image to the new folder with an appropriate name
        if [ -f "$python_output_file" ]; then
            mv "$python_output_file" "$output_file"
        else
            echo "Warning: Plot for $variable ($plot_type) not generated."
        fi
    done
done

echo "All plots have been generated and saved in the folder: $output_folder"
