# CTL(++) policy for simultaneous sign language translation

This is a tutorial of our CTL(++) policy for simultaneous sign language translation.

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

## 2.2. Construct CTL training data

**step 1**: Obtain offline translation results for train/valid/test set and store them.

```bash
bash experiments/obtain_offline_transl_result.sh
```

**step 2**: Construct CTL training data.

```bash
bash experiments/construct_ctl_data.sh
```

> Due to the processing method of SimulEval, it is necessary to clean up some duplicate data.

> Due to the large number of samples in the `train` set, it may lead to slow data read and write speeds. We suggest splitting the training set for processing in **step 1** and **step 2**, then cleaning and merging them at the end.

> An example script for data cleaning is located [here](/experiments/clean_and_merge.py).

If everything goes well, CTL data should look like [this](/experiments/example_ctl_data.json).

## 2.3. Train a CTL estimator

Prepare one config file `sign.yaml` in one directory `configs`. It should look like [this](/experiments/ctl_configs).

Train and evaluate the estimator.

```bash
bash experiments/train_ctl_estimator.sh
```

## 2.4. Adaptive Training with prefix pairs (CTL++)

Adding corresponding prefix pairs from CTL training data to `DATA_BIN` to generate a `NEW_DATA_BIN`. An example script is located [here](experiments/generate_new_data_bin.py).

Using the `NEW_DATA_BIN` to train an offline translation model, and replacing the translation model in CTL with it, resulting in CTL++.

# 3. Evaluation

Evaluate translation quality and latency with a CTL(++) agent.

```
bash experiments/eval_ctl.sh
```
