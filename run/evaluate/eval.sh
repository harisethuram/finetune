dataset=$1
data_dir=$2
val_dataset=$3
val_data_dir=$4
num_labels=$5
b_size=$6
cols=$7
null_lab=$8
classification_idx=-1
models=("facebook/opt-1.3b" "meta-llama/Llama-2-7b-hf")
lambda_entropies=(0.1 0.5 1)
entropy_funcs=("subtract_entropy" "exp_subtract_entropy" "inverse_entropy")
loss_fns=("MASK" "COMP" "CE")

DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"
null_lab=-1
for model in "${models[@]}"; do
    for loss_fn in "${loss_fns[@]}"; do
        for entropy_func in "${entropy_funcs[@]}"; do
            for lambda_entropy in "${lambda_entropies[@]}"; do
                echo "evaluating on $val_dataset/$val_data_dir $model $loss_fn $entropy_func $lambda_entropy"
                python evaluate.py\
                    --finetuned_model_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$model/loss_$loss_fn/$entropy_func/lambda_$lambda_entropy/model.pt"\
                    --model_name $model\
                    --dataset_cache $DATASET_CACHE\
                    --results_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$model/loss_$loss_fn/$entropy_func/lambda_$lambda_entropy/adv_eval/$val_dataset/$val_data_dir/"\
                    --val_dataset $val_dataset\
                    --val_data_dir $val_data_dir\
                    --cols $cols\
                    --split "validation"\
                    --num_labels $num_labels\
                    --null_lab $null_lab\
                    --b_size $b_size\
                    --classification_idx $classification_idx
            done
        done
    done
done
