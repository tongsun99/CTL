#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=2
export SIGN_SEGMENT_SIZE=40 # 40 for PHOENIX2014T, 33 for CSL-Daily
splits=(test valid train)   # split train to train_0, train_1, ... is recommended

for i in "${!splits[@]}"; do
    split=${splits[$i]}
    python ${PATH_TO_SIMULEVAL-SLT}/simuleval/cli.py \
        --agent \
        experiments/agents/mu_data_agent.py \
        --source \
        ${PATH_TO_DATA_BIN}/src_${split}.txt \
        --target \
        ${PATH_TO_DATA_BIN}/tgt_${split}.txt \
        --data-bin \
        ${PATH_TO_DATA_BIN} \
        --model-path \
        ${PATH_TO_OFFLINE_TRANSL_MODEL} \
        --instance-log \
        ${PATH_TO_OFFLINE_RESULT}/${split}/instances.log \
        --output \
        ${PATH_TO_MU_DATA_DIR}/output${split} \
        --output-file \
        ${PATH_TO_MU_DATA_DIR}/${split}.json \
        --scores \
        --remove-subword \
        ${REMOVE_TOKEN_NUM} \
        --port \
        1666${i} \
        --pre-decision-ratio \
        10 \
        --gpu &
done
