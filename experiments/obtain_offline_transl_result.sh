#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=2
export SIGN_SEGMENT_SIZE=40 # 40 for PHOENIX2014T, 33 for CSL-Daily
splits=(test valid train)   # split train to train_0, train_1, ... is recommended

for i in "${!splits[@]}"; do
    split=${splits[$i]}
    python ${PATH_TO_SIMULEVAL-SLT}/simuleval/cli.py \
        --agent \
        experiments/agents/waitinf_agent.py \
        --source \
        ${PATH_TO_DATA_BIN}/src_${split}.txt \
        --target \
        ${PATH_TO_DATA_BIN}/tgt_${split}.txt \
        --model-path \
        ${PATH_TO_OFFLINE_TRANSL_MODEL} \
        --data-bin \
        ${PATH_TO_DATA_BIN} \
        --output \
        ${PATH_TO_OFFLINE_RESULT}/${split} \
        --scores \
        --port \
        1777${i} \
        --gpu \
        --pre-decision-ratio \
        10 &
done
