#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# train
python fairseq_cli/hydra_train.py \
    hydra.run.dir=${PATH_TO_RUN_DIR} \
    hydra.output_subdir=. \
    common.tensorboard_logdir=${PATH_TO_RUN_DIR}/tensorboard \
    common.cpu=False \
    common.seed=1 \
    task.data=${PATH_TO_MU_DATA_DIR} \
    task.data_bin=${PATH_TO_DATA_BIN} \
    optimization.lr=[2e-5] \
    optimization.update_freq=[1] \
    optimization.max_epoch=20 \
    model.pool_method=transformer_cls2 \
    model.classifier_dropout=0.2 \
    model.s2ttransformer_checkpoint=${PATH_TO_OFFLINE_TRANSL_MODEL} \
    dataset.batch_size=64 \
    --config-dir \
    ${PATH_TO_CONFIGS_DIR} \
    --config-name \
    sign

wait

# evaluate
python experiments/eval_sign2vec_cls.py \
    --data \
    ${PATH_TO_MU_DATA_DIR} \
    --subset \
    test \
    --save-dir \
    ${PATH_TO_RUN_DIR} \
    --checkpoint-file \
    checkpoint_best.pt \
    --eval \
    --use-gpu
