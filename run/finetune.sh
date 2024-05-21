dataset="$1"
data_dir="$2"
cols=$3
null_lab=$4
b_size=$5
TRAIN_CAP=$6
VAL_CAP=$7
TEST_CAP=$8
num_labels=$9

num_epochs=1
MODEL_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/"
DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"
# models=("meta-llama/Llama-2-7b-hf")
models=("facebook/opt-1.3b" "meta-llama/Llama-2-7b-hf")
loss_fns=("COMP" "CE")
class_idx=(-1 -1)

for i in "${!models[@]}"; do
    for loss_fn in "${loss_fns[@]}"; do
        m=${models[$i]}
        c=${class_idx[$i]}
        echo "running $m on $dataset/$data_dir w/ $loss_fn"
        python finetune.py\
        --model_name $m\
        --model_cache $MODEL_CACHE\
        --dataset $dataset\
        --data_dir $data_dir\
        --dataset_cache $DATASET_CACHE\
        --results_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$m/$loss_fn/"\
        --num_epochs $num_epochs\
        --null_lab $null_lab\
        --classification_idx $c\
        --b_size $b_size\
        --num_labels $num_labels\
        --train_cap $TRAIN_CAP\
        --val_cap $VAL_CAP\
        --test_cap $TEST_CAP\
        --input_cols "$cols"\
        --loss_fn_name $loss_fn\
        --overwrite
    done
done
