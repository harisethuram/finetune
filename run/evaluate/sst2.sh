dataset="nyu-mll/glue"
data_dir="sst2"
val_dataset="AI-Secure/adv_glue"
val_data_dir="adv_sst2"
num_labels=2
b_size=1
cols="sentence"
null_lab=-2

bash run/evaluate/eval.sh $dataset $data_dir $val_dataset $val_data_dir $num_labels $b_size $cols $null_lab