# Attention is all you need
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch 1.9](https://img.shields.io/badge/pytorch-1.9-orange.svg)](https://pytorch.org/)
[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/Hojjat-Mokhtarabadi/Attention-is-all-you-need)

Pytorch implementation of attention is all you need [paper](https://arxiv.org/abs/1706.03762).
Since the introduction of Transformer architecture, it has become a dominant model in the field of natural language processing achieving remarkable results. The core part of this architecture is 'Attention module' which not only does let the network capture long-range dependencies but also be more parallelizable. However it is shown in some [works](https://arxiv.org/abs/2103.03404) that all parts together achieve superior results and the neglect of some components might lead to degenration of output.

## Requirements
- Python = 3.7
- Pytorch = 1.9

## Install
```bash
git clone https://github.com/Hojjat-Mokhtarabadi/Attention-is-all-you-need.git
cd Attention-is-all-you-need
python3 -m pip install -r requirements.txt
```

## Train
```bash
bash run.sh
```

