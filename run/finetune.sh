dataset="$1"
data_dir="$2"
cols=$3
null_lab=$4
train_b_size=$5
val_b_size=$6
TRAIN_CAP=$7
VAL_CAP=$8
TEST_CAP=$9
num_labels=${10}

num_epochs=1
MODEL_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/"
DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"
models=("facebook/opt-1.3b" "meta-llama/Llama-2-7b-hf")
loss_fns=("CE")
# loss_fns=("COMP")
entropy_funcs=("subtract_entropy" "exp_subtract_entropy")
lambda_entropies=(0)
class_idx=(-1 -1)

for i in "${!models[@]}"; do
    for loss_fn in "${loss_fns[@]}"; do
        for entropy_func in "${entropy_funcs[@]}"; do
            for lambda_entropy in "${lambda_entropies[@]}"; do
                m=${models[$i]}
                c=${class_idx[$i]}
                echo "running $m on $dataset/$data_dir w/ $loss_fn w/ $entropy_func and lamdba ent $lambda_entropy"
                python finetune.py\
                --model_name $m\
                --model_cache $MODEL_CACHE\
                --dataset $dataset\
                --data_dir $data_dir\
                --dataset_cache $DATASET_CACHE\
                --results_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$m/loss_$loss_fn/$entropy_func/lambda_$lambda_entropy/"\
                --num_epochs $num_epochs\
                --null_lab $null_lab\
                --classification_idx $c\
                --train_b_size $train_b_size\
                --val_b_size $val_b_size\
                --num_labels $num_labels\
                --train_cap $TRAIN_CAP\
                --val_cap $VAL_CAP\
                --test_cap $TEST_CAP\
                --input_cols "$cols"\
                --loss_fn_name $loss_fn\
                --entropy_func $entropy_func\
                --lambda_entropy $lambda_entropy\
                --k 5\
                --copy_b_size 16
            done
        done
    done
done
