dataset="stanfordnlp/snli"
data_dir="plain_text"
cols="premise,hypothesis"
null_lab=-1
train_b_size=16
val_b_size=16
train_cap=250000
val_cap=1000000
test_cap=1000
num_labels=3

bash run/finetune.sh $dataset $data_dir $cols $null_lab $train_b_size $val_b_size $train_cap $val_cap $test_cap $num_labels