ROOT="/home/sl1471/workspace/experiments";
data_key="ml-1m";
# data_key="amz_Books";
# data_key="amz_Electronics";
# data_key="amz_Sports_and_Outdoors";
# data_key="amz_Clothing_Shoes_and_Jewelry";
# data_key="amz_Video_Games";

mkdir -p ${ROOT}/${data_key}/models
mkdir -p ${ROOT}/${data_key}/logs

task_name="FairnessEval";
METRIC="_AUC";
device=1;

model_name="FedMF";
LR=0.003;
REG=0.01;
LOSS="pairwisebpr";
NNEG=1;
DIM=32;
DEVICE_DROPOUT=0.1;
N_LOCAL_STEP=1;
ELASTIC_MU=0.01;
FED_TYPE="fedavg";

python main.py\
    --proctitle "Loki"\
    --model ${model_name}\
    --task ${task_name}\
    --user_group_field "Gender" "Age"\
    --loss ${LOSS}\
    --emb_size ${DIM}\
    --eval\
    --cuda ${device}\
    --n_worker 4\
    --model_path ${ROOT}/${data_key}/models/${model_name}_lr${LR}_reg${REG}_loss${LOSS}_local${N_LOCAL_STEP}_${FED_TYPE}.pkl\
    --data_file ${ROOT}/${data_key}/tsv_data/\
    --user_meta_data ${ROOT}/${data_key}/meta_data/user.meta\
    --item_meta_data ${ROOT}/${data_key}/meta_data/item.meta\
    --user_fields_meta_file ${ROOT}/${data_key}/meta_data/user_fields.meta\
    --item_fields_meta_file ${ROOT}/${data_key}/meta_data/item_fields.meta\
    --user_fields_vocab_file ${ROOT}/${data_key}/meta_data/user_fields.vocab\
    --item_fields_vocab_file ${ROOT}/${data_key}/meta_data/item_fields.vocab\
    > ${ROOT}/${data_key}/logs/eval_fairness_${model_name}_lr${LR}_reg${REG}_loss${LOSS}_local${N_LOCAL_STEP}_${FED_TYPE}.log

