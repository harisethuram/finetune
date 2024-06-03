method=$1

dataset="nyu-mll/glue"
data_dir="sst2"
cols="sentence"
null_lab=-2
csv_file="artifact_files/sst2-artifacts.csv"

bash run/generate_plots/two_plot_$method.sh $dataset $data_dir $cols $null_lab $csv_file