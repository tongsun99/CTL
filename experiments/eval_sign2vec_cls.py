import numpy as np
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
        "--eval",
        default=True,
        action="store_true",
        help="eval after inference. save in the same filename with output but .json format. default: True",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for classification. default: 0.5",
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
    ps = []
    gts = []

    with torch.no_grad():
        for iter in tqdm(range(len(data))):
            input = data.__getitem__(iter)
            output = model(
                source=input["source"].unsqueeze(0).to(device), padding_mask=None
            )
            p = torch.nn.functional.softmax(output["pooled"], dim=-1).cpu()[0, 1].item()
            preds.append(int(p > args.threshold))
            ps.append(p)
            gts.append(input["label"])
    id2label = {v: k for k, v in checkpoint.task.label2id.items()}

    output_tsv = os.path.join(args.save_dir, f"pred-{args.subset}-{args.threshold}.sent")
    fid = open(output_tsv, "w")

    for p, pred, gt in zip(ps, preds, gts):
        json_obj = {
            "pred": id2label[pred],
            "gt": id2label[gt],
            "p": f'{p:.2f}',
        }
        fid.write(json.dumps(json_obj) + "\n")
    fid.close()

    if args.eval:
        output_json = os.path.splitext(output_tsv)[0] + ".json"
        json_dict = {}
        json_dict["macro"] = {
            "precision": precision_score(gts, preds, average="macro") * 100,
            "recall": recall_score(gts, preds, average="macro") * 100,
            "f1": f1_score(gts, preds, average="macro") * 100,
        }
        json_dict["micro"] = {
            "precision": precision_score(gts, preds, average="weighted") * 100,
            "recall": recall_score(gts, preds, average="weighted") * 100,
            "f1": f1_score(gts, preds, average="weighted") * 100,
        }
        json_dict["per_classes"] = {
            id2label[idx]: score
            for idx, score in enumerate(f1_score(gts, preds, average=None) * 100)
        }

        with open(output_json, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
