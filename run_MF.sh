ROOT="/home/sl1471/workspace/experiments";

# data_key="ml-1m";
# data_key="amz_Books";
# data_key='amz_Movies_and_TV';
data_key='amz_Electronics';

mkdir -p ${ROOT}/${data_key}/models
mkdir -p ${ROOT}/${data_key}/logs

task_name="TopK";
METRIC="_NDCG@50";
device=1;

model_name="MF";
BS=256;
LOSS="pairwisebpr";
NNEG=1;
DIM=32;


for LR in 0.0001
do
    for REG in 1.0
    do
        python main.py\
            --proctitle "Loki"\
            --model ${model_name}\
            --task ${task_name}\
            --n_round 1\
            --train_and_eval\
            --seed 19\
            --optimizer "Adam"\
            --cuda ${device}\
            --n_worker 4\
            --epoch 30\
            --batch_size ${BS}\
            --lr ${LR}\
            --val_sample_p 1.0\
            --with_val \
            --temper 6\
            --stop_metric ${METRIC}\
            --model_path ${ROOT}/${data_key}/models/f2rec_${model_name}_lr${LR}_reg${REG}_${LOSS}.pkl\
            --loss ${LOSS}\
            --l2_coef ${REG}\
            --emb_size ${DIM}\
            --data_file ${ROOT}/${data_key}/tsv_data/\
            --user_meta_data ${ROOT}/${data_key}/meta_data/user.meta\
            --item_meta_data ${ROOT}/${data_key}/meta_data/item.meta\
            --user_fields_meta_file ${ROOT}/${data_key}/meta_data/user_fields.meta\
            --item_fields_meta_file ${ROOT}/${data_key}/meta_data/item_fields.meta\
            --user_fields_vocab_file ${ROOT}/${data_key}/meta_data/user_fields.vocab\
            --item_fields_vocab_file ${ROOT}/${data_key}/meta_data/item_fields.vocab\
            --n_neg ${NNEG}\
            --n_neg_val 100\
            > ${ROOT}/${data_key}/logs/f2rec_train_and_eval_${model_name}_lr${LR}_reg${REG}_loss${LOSS}.log
    done
done
