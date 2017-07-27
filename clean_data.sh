#!/usr/bin/env bash
. ./set_path.sh
rm -rf ${TRAINDIR} ${TESTDIR} 
mkdir -pv ${TRAINDIR} ${TESTDIR}

python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/train ${TRAINDIR} 
python ${PREPROCESS_DIR}/clean_data.py ${DATA_DIR}/test ${TESTDIR} 
