#!/usr/bin/env bash
. ./set_path.sh
rm -rf ${TRAINDIR_lemma} ${TESTDIR_lemma}
mkdir -pv ${TRAINDIR_lemma} ${TESTDIR_lemma}

python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/train ${TRAINDIR_lemma}
python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/test ${TESTDIR_lemma} 
