#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=1
export SIGN_SEGMENT_SIZE=40 # 40 for PHOENIX2014T, 33 for CSL-Daily

removes=(0 2 4 6 8 10 14 18)

for i in "${!removes[@]}"; do
    remove=${removes[$i]}
    python ${PATH_TO_SIMULEVAL}/simuleval/cli.py \
        --agent \
        experiments/agents/ctl_agent.py \
        --source \
        ${PATH_TO_DATA_BIN}/src_test.txt \
        --target \
        ${PATH_TO_DATA_BIN}/tgt_test.txt \
        --data-bin \
        ${PATH_TO_DATA_BIN} \
        --model-path \
        ${PATH_TO_OFFLINE_TRANSL_MODEL} \
        --ctl-model-path-list \
        ${PATH_TO_CTL_ESTIMATOR_CKPT_1} \
        ${PATH_TO_CTL_ESTIMATOR_CKPT_2} \
        ${PATH_TO_CTL_ESTIMATOR_CKPT_N} \
        --remove-subword \
        ${remove} \
        --output \
        ${PATH_TO_OUTPUT_DIR}/remove${remove} \
        --port \
        1555${i} \
        --scores \
        --pre-decision-ratio \
        10 \
        --gpu &
done
