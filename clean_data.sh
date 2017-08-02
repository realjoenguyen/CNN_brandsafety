#!/usr/bin/env bash
. ./set_path.sh
rm -rf ${TRAINDIR_lemma_extensive} ${TESTDIR_lemma_extensive}
mkdir -pv ${TRAINDIR_lemma_extensive} ${TESTDIR_lemma_extensive}

python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/train ${TRAINDIR_lemma_extensive} 
python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/test ${TESTDIR_lemma_extensive} 
