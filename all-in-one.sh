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
        --trainVocab False \
        --exp_name newest > train-log.txt

#python ${CODE_DIR}/eval.py \
#        --testdir ${TESTDIR} \
#        --checkpoint_dir='./runs/newest/checkpoints/'