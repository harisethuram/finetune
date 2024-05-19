dataset="stanfordnlp/snli"
data_dir="plain_text"
cols="premise,hypothesis"
null_lab=-1
b_size=64
train_cap=100
val_cap=1000
test_cap=1000
num_labels=3

bash run/finetune.sh $dataset $data_dir $cols $null_lab $b_size $train_cap $val_cap $test_cap $num_labels