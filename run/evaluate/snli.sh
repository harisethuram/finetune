dataset="stanfordnlp/snli"
data_dir="plain_text"
val_dataset="AI-Secure/adv_glue"
val_data_dir="adv_mnli"
num_labels=3
b_size=1
cols="premise,hypothesis"
null_lab=-1

bash run/evaluate/eval.sh $dataset $data_dir $val_dataset $val_data_dir $num_labels $b_size $cols $null_lab