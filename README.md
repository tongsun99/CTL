# Adaptive Simultaneous Sign Language Translation with Confident Translation Length Estimation
Source code for our LREC-COLING2024 paper ["Adaptive Simultaneous Sign Language Translation with Confident Translation Length Estimation"](https://aclanthology.org/2024.lrec-main.34/).

We also provide [SimulEval-SLT](https://github.com/tongsun99/SimulEval-SLT), a general evaluation framework for simultaneous translation on sign languages.

# Requirements and Installation

Install fairseq and develop locally

```bash
git clone https://github.com/tongsun99/CTL
cd CTL
pip install -e .
```

Install SimulEval-SLT and develop locally

```bash
git clone https://github.com/tongsun99/SimulEval-SLT
cd SimulEval-SLT
pip install -e .
```

# Usage

Detailed introductions refer to:

- [Wait-k](./README-waitk.md)
- [MU-ST](./README-must.md)
- [CTL(++)](./README-ctl(++).md)

# Acknowledgements

Our code is inspired by the following repositories. Many thanks to their work!

- [SimulEval](https://github.com/facebookresearch/SimulEval)

- [Fairseq](https://github.com/facebookresearch/fairseq)

- [slue-toolkit](https://github.com/asappresearch/slue-toolkit)

# Citation

```
@inproceedings{sun2024adaptive,
  title={Adaptive Simultaneous Sign Language Translation with Confident Translation Length Estimation},
  author={Sun, Tong and Fu, Biao and Hu, Cong and Zhang, Liang and Zhang, Ruiquan and Shi, Xiaodong and Su, Jinsong and Chen, Yidong},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={372--384},
  year={2024}
}
```