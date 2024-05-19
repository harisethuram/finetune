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
# FINETUNED_MODEL_DIR=f"/gscratch/ark/hari/msc/results/{DATASET}/{DATA_DIR}/{MODEL_NAME}/model.pt"
# models=("meta-llama/Llama-2-7b-hf")
models=("facebook/opt-1.3b") # "meta-llama/Llama-2-7b-hf")
class_idx=(-1 -1)

for i in "${!models[@]}"; do
    
    m=${models[$i]}
    c=${class_idx[$i]}
    echo "running $m on $dataset/$data_dir"
    python finetune.py\
    $m\
    $MODEL_CACHE\
    $dataset\
    $data_dir\
    $DATASET_CACHE\
    "/gscratch/ark/hari/msc/tests/embs_tests/results/$dataset/$data_dir/$m/"\
    "/gscratch/ark/hari/msc/tests/embs_tests/results/$dataset/$data_dir/$m/model.pt"\
    $num_epochs\
    $null_lab\
    $c\
    $b_size\
    $num_labels\
    $TRAIN_CAP\
    $VAL_CAP\
    $TEST_CAP\
    "$cols"
done
