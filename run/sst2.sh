dataset="nyu-mll/glue"
data_dir="sst2"
cols="sentence"
null_lab=-2
train_b_size=16
val_b_size=64
train_cap=100000
val_cap=1000
test_cap=1000
num_labels=2

bash run/finetune.sh $dataset $data_dir $cols $null_lab $train_b_size $val_b_size $train_cap $val_cap $test_cap $num_labels