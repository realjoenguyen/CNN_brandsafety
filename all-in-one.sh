#!/bin/bash -x
# Step 1: Prepare text files:
#build vocab
. ./set_path.sh
rm -rf ${DATA_OWN}
mkdir -pv ${DATA_OWN}

#build vocab
#python ${PREPROCESS_DIR}/build_vocab.py ${TRAINDIR} ${DATA_OWN}/vocab.pkl
#for each file, only consider the word in new vocab, discard the others
#rm -rf ${TRAINDIR_REDUCED} ${TESTDIR_REDUCED}
#mkdir -pv ${TRAINDIR_REDUCED} ${TESTDIR_REDUCED}
#python ${PREPROCESS_DIR}/reduce_vocab.py ${TRAINDIR} ${TRAINDIR_REDUCED} ${DATA_OWN}/vocab.pkl 20000
#python ${PREPROCESS_DIR}/reduce_vocab.py ${TESTDIR} ${TESTDIR_REDUCED} ${DATA_OWN}/vocab.pkl 20000

python ${CODE_DIR}/train.py \
        --traindir ${TRAINDIR} \
        --pretrain True \
        --pretrain_data word2vec \
        --trainVocab True \
        --exp_name _100 \
        --init_word2vec_random False \
        --batch_size 40 \`
        --L2 3 \
        --num_filters 100 \
        --dropout_keep_prob 0.5 \
        --num_epochs 10 > train-log-_100.txt \
&& python ${CODE_DIR}/eval.py \
        --traindir ${TRAINDIR} \
        --testdir ${TESTDIR} \
        --dev True \
        --checkpoint_dir='./runs/_100/checkpoints/' > test-log-_100.txt \
&& rm -rf './runs/_100/checkpoints/'

python ${CODE_DIR}/train.py \
        --traindir ${TRAINDIR} \
        --pretrain True \
        --pretrain_data word2vec \
        --trainVocab True \
        --exp_name _128 \
        --init_word2vec_random False \
        --batch_size 40 \`
        --L2 3 \
        --num_filters 128 \
        --dropout_keep_prob 0.5 \
        --num_epochs 10 > train-log-_128.txt \
&& python ${CODE_DIR}/eval.py \
        --traindir ${TRAINDIR} \
        --testdir ${TESTDIR} \
        --dev True \
        --checkpoint_dir='./runs/_128/checkpoints/' > test-log-_128.txt \
&& rm -rf './runs/_128/checkpoints/'

#
#python ${CODE_DIR}/train.py \
#        --traindir ${TRAINDIR} \
#        --pretrain True \
#        --pretrain_data word2vec \
#        --trainVocab False \
#        --exp_name _200 \
#        --init_word2vec_random False \
#        --batch_size 40 \
#        --L2 3 \
#        --num_filters 200 \
#        --dropout_keep_prob 0.5 \
#        --num_epochs 10 > train-log-_200.txt \
#&& python ${CODE_DIR}/eval.py \
#        --traindir ${TRAINDIR} \
#        --testdir ${TESTDIR} \
#        --dev True \
#        --checkpoint_dir='./runs/_200/checkpoints/' > test-log-_200.txt
#
#python ${CODE_DIR}/train.py \
#        --traindir ${TRAINDIR} \
#        --pretrain True \
#        --pretrain_data word2vec \
#        --trainVocab False \
#        --exp_name _300 \
#        --init_word2vec_random False \
#        --batch_size 40 \
#        --L2 3 \
#        --num_filters 300 \
#        --dropout_keep_prob 0.5 \
#        --num_epochs 10 > train-log-_300.txt \
#&& python ${CODE_DIR}/eval.py \
#        --traindir ${TRAINDIR} \
#        --testdir ${TESTDIR} \
#        --dev True \
#        --checkpoint_dir='./runs/_300/checkpoints/' > test-log-_300.txt

python ${CODE_DIR}/train.py \
        --traindir ${TRAINDIR} \
        --pretrain True \
        --pretrain_data word2vec \
        --trainVocab False \
        --exp_name _400 \
        --init_word2vec_random False \
        --batch_size 40 \
        --L2 3 \
        --num_filters 400 \
        --dropout_keep_prob 0.5 \
        --num_epochs 10 > train-log-_400.txt \
&& python ${CODE_DIR}/eval.py \
        --traindir ${TRAINDIR} \
        --testdir ${TESTDIR} \
        --dev True \
        --checkpoint_dir='./runs/_400/checkpoints/' > test-log-_400.txt \
&& rm -rf './runs/_400/checkpoints/'

python ${CODE_DIR}/train.py \
        --traindir ${TRAINDIR} \
        --pretrain True \
        --pretrain_data word2vec \
        --trainVocab False \
        --exp_name _500 \
        --init_word2vec_random False \
        --batch_size 40 \
        --L2 3 \
        --num_filters 500 \
        --dropout_keep_prob 0.5 \
        --num_epochs 10 > train-log-_500.txt \
&& python ${CODE_DIR}/eval.py \
        --traindir ${TRAINDIR} \
        --testdir ${TESTDIR} \
        --dev True \
        --checkpoint_dir='./runs/_500/checkpoints/' > test-log-_500.txt \
&& rm -rf './runs/_500/checkpoints/'


python ${CODE_DIR}/train.py \
        --traindir ${TRAINDIR} \
        --pretrain True \
        --pretrain_data word2vec \
        --trainVocab False \
        --exp_name _600 \
        --init_word2vec_random False \
        --batch_size 40 \
        --L2 3 \
        --num_filters 600 \
        --dropout_keep_prob 0.5 \
        --num_epochs 10 > train-log-_600.txt \
&& python ${CODE_DIR}/eval.py \
        --traindir ${TRAINDIR} \
        --testdir ${TESTDIR} \
        --dev True \
        --checkpoint_dir='./runs/_600/checkpoints/' > test-log-_600.txt \
&& rm -rf './runs/_600/checkpoints/'

