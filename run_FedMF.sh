ROOT="/home/sl1471/workspace/experiments";

data_key="ml-1m"; 
# data_key="amz_Books";
# data_key="amz_Electronics";
# data_key='BX'; #
# data_key='amz_Movies_and_TV';

# train_file=${ROOT}${data_key}"/tsv_data/train.tsv";
# val_file=${ROOT}${data_key}"/tsv_data/val.tsv";
# test_file=${ROOT}${data_key}"/tsv_data/test.tsv";
# user_vocab=${ROOT}${data_key}"/meta_data/user.vocab";
# item_vocab=${ROOT}${data_key}"/meta_data/item.vocab";

mkdir -p ${ROOT}/${data_key}/models
mkdir -p ${ROOT}/${data_key}/logs

task_name="FedTopK";
METRIC="_AUC";
device=1;

model_name="FedMF";
LOSS="pairwisebpr";
NNEG=1;
DIM=32;
DEVICE_DROPOUT=0.1;
ELASTIC_MU=0.01;
FED_TYPE="fedavg";
FED_BETA=1.0;

for REG in 0.1 1.0
do
    for N_LOCAL_STEP in 1 3 5 #7 5
    do
        for LR in 0.003 0.001 0.01 
        do
            python main.py\
                --proctitle "Baldr"\
                --model ${model_name}\
                --task ${task_name}\
                --n_round 1\
                --train_and_eval\
                --seed 19\
                --optimizer "SGD"\
                --cuda ${device}\
                --n_worker 4\
                --epoch 40\
                --lr ${LR}\
                --val_sample_p 0.5\
                --with_val \
                --temper 6\
                --stop_metric ${METRIC}\
                --model_path ${ROOT}/${data_key}/models/f2rec_${model_name}_lr${LR}_reg${REG}_${LOSS}_local${N_LOCAL_STEP}_${FED_TYPE}.pkl\
                --loss ${LOSS}\
                --l2_coef ${REG}\
                --emb_size ${DIM}\
                --device_dropout_p ${DEVICE_DROPOUT}\
                --n_local_step ${N_LOCAL_STEP}\
                --aggregation_func ${FED_TYPE}\
                --mitigation_trade_off ${FED_BETA}\
                --elastic_mu ${ELASTIC_MU}\
                --data_file ${ROOT}/${data_key}/tsv_data/\
                --user_meta_data ${ROOT}/${data_key}/meta_data/user.meta\
                --item_meta_data ${ROOT}/${data_key}/meta_data/item.meta\
                --user_fields_meta_file ${ROOT}/${data_key}/meta_data/user_fields.meta\
                --item_fields_meta_file ${ROOT}/${data_key}/meta_data/item_fields.meta\
                --user_fields_vocab_file ${ROOT}/${data_key}/meta_data/user_fields.vocab\
                --item_fields_vocab_file ${ROOT}/${data_key}/meta_data/item_fields.vocab\
                --n_neg ${NNEG}\
                --n_neg_val 100\
                > ${ROOT}/${data_key}/logs/f2rec_train_and_eval_${model_name}_lr${LR}_reg${REG}_loss${LOSS}_local${N_LOCAL_STEP}_${FED_TYPE}.log
        done
    done
done