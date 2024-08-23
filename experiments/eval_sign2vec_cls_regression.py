import numpy as np
import math
import torch, argparse, json, os
from tqdm import tqdm
from torch import cuda
from fairseq.dataclass import FairseqDataclass
from fairseq.models.sign_classification.sign2vec_cls import Sign2VecSeqCls
from fairseq.tasks.sign_classification import SignClassificationTask
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="manifest dir for data loading"
    )
    parser.add_argument(
        "--subset", type=str, required=True, help="split name (test, dev, finetune"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="save dir containing checkpoints folder",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=False,
        default="checkpoint_best.pt",
        help=".pt file you want to use",
    )
    parser.add_argument(
        "--use-gpu",
        default=False,
        action="store_true",
        help="use gpu if available default: False",
    )
    parser.add_argument(
        "--random",
        default=False,
        type=bool,
        help="random prediction. default: False",
    )
    args = parser.parse_args()
    print(args)

    if args.use_gpu:
        device = "cuda:0" if cuda.is_available() else "cpu"
    else:
        device = "cpu"
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    checkpoint = Sign2VecSeqCls.from_pretrained(
        checkpoint_dir, checkpoint_file=args.checkpoint_file
    )

    checkpoint.task.cfg.data = args.data
    checkpoint.task.load_dataset(args.subset)
    checkpoint.task.load_label2id
    checkpoint.to(device)
    data = checkpoint.task.datasets[args.subset]
    model = checkpoint.models[0]
    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for iter in tqdm(range(len(data))):
            input = data.__getitem__(iter)
            output = model(
                source=input["source"].unsqueeze(0).to(device), padding_mask=None
            )
            if checkpoint.task.cfg.label_norm == "log":
                pred = output["pooled"].item()
                pred = int(round(math.exp(pred)))
                gt = math.exp(input["label"])
                preds.append(pred)
                gts.append(gt)
            elif checkpoint.task.cfg.label_norm == "min_max":
                pred = output["pooled"].item()
                pred = int(round(pred * (checkpoint.task.cfg.max_len - checkpoint.task.cfg.min_len) + checkpoint.task.cfg.min_len))
                gt = int(round(input["label"] * (checkpoint.task.cfg.max_len - checkpoint.task.cfg.min_len) + checkpoint.task.cfg.min_len))
                preds.append(pred)
                gts.append(gt)
            else:
                pred = output["pooled"].item()
                pred = int(round(pred))
                gt = int(round(input["label"]))
                preds.append(pred)
                gts.append(gt)

    id2label = {v: k for k, v in checkpoint.task.label2id.items()}

    mse_loss = 0.0
    with open(os.path.join(args.save_dir, f"pred-{args.subset}.json"), "w+") as f:
        for pred, gt in zip(preds, gts):
            pred = max(pred, 0)
            f.write(json.dumps({"pred": pred, "gt": gt}) + "\n")
            mse_loss += (pred - gt) ** 2
    mse_loss /= len(preds)
    with open(os.path.join(args.save_dir, f"metric-{args.subset}.json"), "w+") as f:
        f.write(json.dumps({"mse": mse_loss}) + "\n")

if __name__ == "__main__":
    main()
