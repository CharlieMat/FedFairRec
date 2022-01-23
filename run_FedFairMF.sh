ROOT="/home/sl1471/workspace/experiments";

# data_key="ml-1m";
# data_key="amz_Books";
data_key='amz_Movies_and_TV';

# train_file=${ROOT}${data_key}"/tsv_data/train.tsv";
# val_file=${ROOT}${data_key}"/tsv_data/val.tsv";
# test_file=${ROOT}${data_key}"/tsv_data/test.tsv";
# user_vocab=${ROOT}${data_key}"/meta_data/user.vocab";
# item_vocab=${ROOT}${data_key}"/meta_data/item.vocab";

mkdir -p ${ROOT}/${data_key}/models
mkdir -p ${ROOT}/${data_key}/logs

task_name="FedFairTopK";
METRIC="_NDCG@50";
device=1;

model_name="FedMF";
REG=1.0;
LOSS="pairwisebpr";
NNEG=1;
DIM=32;

rho=1;
group='activity';

DEVICE_DROPOUT=0.1;
ELASTIC_MU=0.01;
FED_TYPE="fedavg";
FED_BETA=1.0;
N_LOCAL_STEP=1;

for LR in 0.003
do
    for sigma in 0.001
    do
        for lambda in 0.1 0.3 0.5 0.7 0.9 -0.1 -0.3 -0.5 -0.7
        do
            python main.py\
                --proctitle "Freyr"\
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
                --val_sample_p 1.0\
                --with_val \
                --temper 6\
                --stop_metric ${METRIC}\
                --model_path ${ROOT}/${data_key}/models/f2rec_Fair${model_name}_lr${LR}_reg${REG}_loss${LOSS}_lambda${lambda}_sigma${sigma}_g${group}.pkl\
                --loss ${LOSS}\
                --l2_coef ${REG}\
                --emb_size ${DIM}\
                --fair_rho ${rho}\
                --fair_lambda ${lambda}\
                --fair_group_feature ${group}\
                --fair_noise_sigma ${sigma}\
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
                > ${ROOT}/${data_key}/logs/f2rec_train_and_eval_Fair${model_name}_lr${LR}_reg${REG}_loss${LOSS}_lambda${lambda}_sigma${sigma}_g${group}.log
        done
    done
done
