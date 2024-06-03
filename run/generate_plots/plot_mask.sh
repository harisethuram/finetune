dataset=$1
data_dir=$2
cols=$3
null_lab=$4
csv_file=$5

models=("facebook/opt-1.3b")
lambda_entropies=(0.1 0.5 1)
entropy_funcs=("subtract_entropy" "exp_subtract_entropy" "inverse_entropy")
loss_fns=("MASK" "COMP" "CE")

DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"
null_lab=-1
for model in "${models[@]}"; do
    for loss_fn in "${loss_fns[@]}"; do
        for entropy_func in "${entropy_funcs[@]}"; do
            for lambda_entropy in "${lambda_entropies[@]}"; do
                echo "masked artifacts $csv_file"
                python artifact_effects_masked.py\
                    --finetuned_model_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$model/loss_$loss_fn/$entropy_func/lambda_$lambda_entropy/model.pt"\
                    --model_name $model\
                    --results_dir "/gscratch/ark/hari/finetune/results/$dataset/$data_dir/$model/loss_$loss_fn/$entropy_func/lambda_$lambda_entropy/"\
                    --dataset $dataset\
                    --data_dir $data_dir\
                    --dataset_cache $DATASET_CACHE\
                    --split "validation"\
                    --cols $cols\
                    --null_lab $null_lab\
                    --csv_file $csv_file
            done
        done
    done
done
# python artifact_effects_masked.py\
#     --finetuned_model_dir "/gscratch/ark/hari/finetune/results/stanfordnlp/snli/plain_text/facebook/opt-1.3b/loss_MASK/exp_subtract_entropy/lambda_0.1/model.pt"\
#     --model_name "facebook/opt-1.3b"\
#     --results_dir "/gscratch/ark/hari/finetune/results/stanfordnlp/snli/plain_text/facebook/opt-1.3b/loss_MASK/exp_subtract_entropy/lambda_0.1/masked_plt.png"\
#     --dataset "stanfordnlp/snli"\
#     --data_dir "plain_text"\
#     --dataset_cache $DATASET_CACHE\
#     --split "validation"\
#     --cols "premise,hypothesis"\
#     --null_lab $null_lab
#     --csv_file artifact_files/snli_artifacts.csv