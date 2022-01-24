ROOT="~/workspace/experiments";

data_key="ml-1m";
# data_key="amz_Movies_and_TV";

task_name="FairTopK";
METRIC="_NDCG@50";
device=0; # -1 if using cpu

model_name="MF";
REG=0.1;
LOSS="pairwisebpr";
NNEG=1;
DIM=32;

rho=1;

for LR in 0.00003 0.0001 0.00001
do
    for group in 'activity' 'Gender' 'Age'
    do
        for lambda in 0.1 0.3 0.5 0.7 0.9 -0.1 -0.3 -0.5 -0.7
        do
            python main.py\
                --proctitle "Thor"\
                --model ${model_name}\
                --task ${task_name}\
                --n_round 1\
                --train_and_eval\
                --seed 29\
                --optimizer "Adam"\
                --cuda ${device}\
                --n_worker 4\
                --epoch 30\
                --lr ${LR}\
                --val_sample_p 1.0\
                --with_val \
                --temper 6\
                --stop_metric ${METRIC}\
                --model_path ${ROOT}/${data_key}/models/f2rec_Fair${model_name}_lr${LR}_reg${REG}_${LOSS}_lambda${lambda}_g${group}.pkl\
                --loss ${LOSS}\
                --l2_coef ${REG}\
                --emb_size ${DIM}\
                --fair_rho ${rho}\
                --fair_lambda ${lambda}\
                --fair_group_feature ${group}\
                --data_file ${ROOT}/${data_key}/tsv_data/\
                --user_meta_data ${ROOT}/${data_key}/meta_data/user.meta\
                --item_meta_data ${ROOT}/${data_key}/meta_data/item.meta\
                --user_fields_meta_file ${ROOT}/${data_key}/meta_data/user_fields.meta\
                --item_fields_meta_file ${ROOT}/${data_key}/meta_data/item_fields.meta\
                --user_fields_vocab_file ${ROOT}/${data_key}/meta_data/user_fields.vocab\
                --item_fields_vocab_file ${ROOT}/${data_key}/meta_data/item_fields.vocab\
                --n_neg ${NNEG}\
                --n_neg_val 100\
                > ${ROOT}/${data_key}/logs/f2rec_train_and_eval_Fair${model_name}_lr${LR}_reg${REG}_loss${LOSS}_lambda${lambda}_g${group}.log
        done
    done
done
