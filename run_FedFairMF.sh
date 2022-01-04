ROOT="/home/sl1471/workspace/experiments";
data_key="ml-1m";
# data_key="amz_Books";
# data_key="amz_Electronics";
# data_key="amz_Sports_and_Outdoors";
# data_key="amz_Clothing_Shoes_and_Jewelry";
# data_key="amz_Video_Games";

# train_file=${ROOT}${data_key}"/tsv_data/train.tsv";
# val_file=${ROOT}${data_key}"/tsv_data/val.tsv";
# test_file=${ROOT}${data_key}"/tsv_data/test.tsv";
# user_vocab=${ROOT}${data_key}"/meta_data/user.vocab";
# item_vocab=${ROOT}${data_key}"/meta_data/item.vocab";

mkdir -p ${ROOT}/${data_key}/models
mkdir -p ${ROOT}/${data_key}/logs

task_name="FedFairTopK";
METRIC="_AUC";
device=2;

model_name="FedFairMF";
LR=0.01;
REG=0.01;
LOSS="pairwisebpr";
NNEG=1;
DIM=32;

rho=2;
lambda=0.1;
group='activity';

DEVICE_DROPOUT=0.1;
ELASTIC_MU=0.01;
FED_TYPE="fedavg";
N_LOCAL_STEP=5;

for lambda in 0.1 #0.3 1.0 3.0 0.
do
    python main.py\
        --proctitle "Odin"\
        --model ${model_name}\
        --task ${task_name}\
        --n_round 3\
        --train_and_eval\
        --seed 9\
        --optimizer "SGD"\
        --cuda ${device}\
        --n_worker 4\
        --epoch 30\
        --batch_size 512\
        --lr ${LR}\
        --val_sample_p 0.1\
        --with_val \
        --temper 6\
        --stop_metric ${METRIC}\
        --model_path ${ROOT}/${data_key}/models/${model_name}_lr${LR}_reg${REG}_${LOSS}.pkl\
        --loss ${LOSS}\
        --l2_coef ${REG}\
        --emb_size ${DIM}\
        --fair_rho ${rho}\
        --fair_lambda ${lambda}\
        --fair_group_feature ${group}\
        --device_dropout_p ${DEVICE_DROPOUT}\
        --n_local_step ${N_LOCAL_STEP}\
        --aggregation_func ${FED_TYPE}\
        --elastic_mu ${ELASTIC_MU}\
        --data_file ${ROOT}/${data_key}/tsv_data/\
        --user_meta_data ${ROOT}/${data_key}/meta_data/user.meta\
        --item_meta_data ${ROOT}/${data_key}/meta_data/item.meta\
        --user_fields_meta_file ${ROOT}/${data_key}/meta_data/user_fields.meta\
        --item_fields_meta_file ${ROOT}/${data_key}/meta_data/item_fields.meta\
        --user_fields_vocab_file ${ROOT}/${data_key}/meta_data/user_fields.vocab\
        --item_fields_vocab_file ${ROOT}/${data_key}/meta_data/item_fields.vocab\
        --n_neg ${NNEG}\
        --n_neg_val 100
#         > ${ROOT}/${data_key}/logs/fairtrain_and_eval_${model_name}_lr${LR}_reg${REG}_loss${LOSS}_lambda${lambda}.log
done
