method=$1

dataset="stanfordnlp/snli"
data_dir="plain_text"
cols="premise,hypothesis"
null_lab=-1
csv_file="artifact_files/snli-artifacts.csv"

bash run/generate_plots/plot_$method.sh $dataset $data_dir $cols $null_lab $csv_file