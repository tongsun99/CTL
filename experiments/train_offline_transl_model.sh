#!/bin/bash
# shellcheck disable=SC2086
export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/train.py ${PATH_TO_DATA_BIN} \
        --task sign_to_gloss_and_text \
        --save-dir ${PATH_TO_SAVE_DIR} \
        --criterion label_smoothed_cross_entropy_with_ctc \
        --arch sign2text_transformer \
        --config-yaml config.yaml \
        --batch-size 16 --max-epoch ${MAX_EPOCH} \
        --optimizer adam --adam-eps 1e-03 --adam-betas '(0.9,0.98)' --clip-norm 10.0 \
        --lr-scheduler inverse_sqrt --lr 1e-3 --warmup-init-lr 1e-7 --warmup-updates 5000 \
        --share-ctc-and-embed \
        --weight-decay 0.0001 \
        --label-smoothing 0.2 \
        --dropout 0.2 \
        --attention-dropout 0.2 \
        --activation-dropout 0.2 \
        --encoder-layers 3 \
        --decoder-layers 3 \
        --encoder-no-scale-embedding \
        --share-decoder-input-output-embed \
        --subsampling-type conv1d --subsampling-layers 1 --subsampling-filter 512 --subsampling-kernel 5 --subsampling-stride 2 \
        --subsampling-norm none \
        --subsampling-activation glu \
        --activation-fn relu \
        --post-process sentencepiece \
        --eval-bleu --eval-bleu-detok space --eval-bleu-remove-bpe sentencepiece --eval-bleu-print-samples \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --scoring sacrebleu \
        --no-epoch-checkpoints \
        --keep-best-checkpoints 10 \
        --distributed-world-size 1 \
        --num-workers 1 \
        --no-progress-bar \
        --update-freq 2 \
        --log-format simple \
        --log-interval 2 \
        --seed 42 \
        --report-accuracy \
        > ${PATH_TO_TRAIN_LOG}