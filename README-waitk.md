# Wait-k policy for simultaneous sign language translation

This is a tutorial of [Wait-k](https://aclanthology.org/P19-1289/) policy for simultaneous sign language translation.

# 1. Data Preparation

The raw data are from:
- [PHOENIX2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [CSL-daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

We provide pre-extracted 2D visual features for [PHOENIX2014T]() and [CSL-Daily](). We ensure that the feature length and the number of video frames are the same for each video. 

| Dataset | FPS | Frame Duration |
|:-------:|:---:|:--------------:|
| PHOENIX2014T | 25 | 40ms |
| CSL-Daily | 30 | 33ms |

Generate auxiliary files for train/valid/test set:

```bash
python experiments/gen_src_tgt_txt.py \
    --data_bin ${ABS_PATH_TO_DATA_BIN} \
    --split ${SPLIT} \
    --sample_rate ${SAMPLE_RATE} # 25 for PHOENIX2014T, 30 for CSL-Daily
```

Build bpe vocabulary:

```bash
python experiments/build_vocab.py \
    --data_bin ${ABS_PATH_TO_DATA_BIN} \
    --vocab_size ${VOCAB_SIZE}
```

Create a `config.yaml` file:
```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ABS_PATH_TO_BPE_MODEL
input_channels: 1024    # 1024 for PHOENIX2014T, 512 for CSL-Daily
input_feat_per_channel: 1
vocab_filename: ABS_PATH_TO_BPE_VOCAB
```

Finally, the directory `DATA_BIN` should look like:

```
├── train
├── valid
├── test
├── mypiece_4500.model
├── mypiece_4500.vocab
├── config_4500.yaml
├── src_train.txt
├── src_valid.txt
├── src_test.txt
├── tgt_train.txt
├── tgt_valid.txt
├── tgt_test.txt
```

# 2. Training
## 2.1. Train an offline sign language translation model

```bash
bash experiments/train_offline_transl_model.sh
```

# 3. Evaluation

Evaluate translation quality and latency with a wait-k agent.

```
bash experiments/eval_waitk.sh
```
