#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=0
# MIN_LEN is the min label + 1 in train.json of CTL_DATA
# MAX_LEN is the max label + 1 in train.json of CTL_DATA
# In our CTL data, the MAX_LEN/MIN_LEN are 90/1 for PHOENIX2014T, 69/1 for CSL-Daily
# During the inference stage, we trained five estimators with five seeds, 
# using the average of their predicted lengths as the result at each step.

# train
python fairseq_cli/hydra_train.py \
    hydra.run.dir=${PATH_TO_RUN_DIR} \
    hydra.output_subdir=. \
    common.tensorboard_logdir=${PATH_TO_RUN_DIR}/tensorboard \
    common.cpu=False \
    task.data=${PATH_TO_CTL_DATA_DIR} \
    task.data_bin=${PATH_TO_DATA_BIN} \
    task.label_norm=min_max \
    task.min_len=${MIN_LEN} \
    task.max_len=${MAX_LEN} \
    common.seed=${SEED} \
    optimization.lr=[2e-5] \
    optimization.update_freq=[1] \
    optimization.max_epoch=20 \
    model.pool_method=transformer_cls2 \
    model.classifier_dropout=0.2 \
    model.freeze_encoder=True \
    model.s2ttransformer_checkpoint=${PATH_TO_OFFLINE_TRANSL_MODEL} \
    criterion.loss_fn=custom_mse \
    dataset.batch_size=32 \
    --config-dir \
    ${PATH_TO_CONFIGS_DIR} \
    --config-name \
    sign

# evaluate
python experiments/eval_sign2vec_cls_regression.py \
    --data \
    ${PATH_TO_CTL_DATA_DIR} \
    --subset \
    test \
    --save-dir \
    ${PATH_TO_RUN_DIR} \
    --checkpoint-file \
    checkpoint_best.pt \
    --eval \
    --use-gpu
