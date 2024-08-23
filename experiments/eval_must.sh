#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=1
export SIGN_SEGMENT_SIZE=40

ths=(0.3 0.5 0.7 0.8 0.85 0.9 0.95 1.0)

for i in "${!ths[@]}"; do
    th=${ths[$i]}
    python ${PATH_TO_SIMULEVAL-SLT}/simuleval/cli.py \
        --agent \
        experiments/agents/mu_agent.py \
        --source \
        ${PATH_TO_DATA_BIN}/src_test.txt \
        --target \
        ${PATH_TO_DATA_BIN}/tgt_test.txt \
        --data-bin \
        ${PATH_TO_DATA_BIN} \
        --model-path \
        ${PATH_TO_OFFLINE_TRANSL_MODEL} \
        --mu-model-path \
        ${PATH_TO_MU_CLASSFIER_CKPT} \
        --mu-threshold \
        ${th} \
        --output \
        ${PATH_TO_OUTPUT_DIR}/th${th} \
        --port \
        1888${i} \
        --scores \
        --remove-subword \
        ${REMOVE_TOKEN_NUM} \
        --pre-decision-ratio \
        10 \
        --gpu &
done
