#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=0
export SIGN_SEGMENT_SIZE=40 # 40 for PHOENIX2014T, 33 for CSL-Daily

ks=(1 3 5 7 9 11)

for i in "${!ks[@]}"; do
    k=${ks[$i]}
    python ${PATH_TO_SIMULEVAL-SLT}/simuleval/cli.py \
        --agent \
        experiments/agents/waitk_agent.py \
        --source \
        ${PATH_TO_DATA_BIN}/src_test.txt \
        --target \
        ${PATH_TO_DATA_BIN}/tgt_test.txt \
        --model-path \
        ${PATH_TO_OFFLINE_TRANSL_MODEL} \
        --config \
        ${PATH_TO_CONFIG_YAML} \
        --data-bin \
        ${PATH_TO_DATA_BIN} \
        --waitk \
        ${k} \
        --output \
        ${PATH_TO_OUTPUT_DIR}/wait${k} \
        --scores \
        --port \
        1566${i} \
        --gpu \
        --pre-decision-ratio \
        10 &
done
