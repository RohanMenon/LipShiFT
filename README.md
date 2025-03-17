# LipShiFT: A Certifiably Robust Shift-based Vision Transformer


This repository provides the implementation of our certifiably robust vision transformer. The code presented here is part of a paper accepted to the ICLR 2025 Workshop ([VerifAI](https://iclr.cc/virtual/2025/workshop/23973)).

The codebase is adapted from the work on LiResNet ([NeurIPS 2023](https://arxiv.org/abs/2301.12549)) and CHORD-LiResNet ([ICLR 2024](https://openreview.net/forum?id=qz3mcn99cu))


## Getting Started:

### Creating a development environment:
Install the latest [Anaconda distribution](https://www.anaconda.com/docs/getting-started/anaconda/install) into your local/remote session.

The following command creates a viable environment using the `environment.yaml` which has all the dependencies required.

```conda
conda env create -f environment.yml
```

Now activate the environment using the following command:

```conda
conda activate lipshift
```

### Training and Evaluation:
Now you can follow along the code and custom configs as described below:
1. For training a model, check out our `run.sh` as a starting point. 
2. Dive into our [configs](/configs) for additional dataset configurations.

Relevant scripts:
1. `train_lipshift.py`: script to train custom LipShiFT model on a specified dataset.
2. `test_script.py`:  Evaluation of a trained checkpoint on a specified dataset. 
3. `eval.py`: Evaluation of Empirical robustness for a trained checkpoint using AutoAttack framework.

Steps to reproduce results:
1. Update config file for LipShiFT under `/configs/lipshift` for the specific dataset.
2. Run command for the same dataset in the `run.sh` bash file.
3. Output checkpoint will be saved in `/checkpoints_all`.

To train with additional augmented data:
1. Create `/data` and store the zipped dataset there. The augmented loader expects a `.npz` file with `dict` keys as `image` and `label`. The images are stored as a list of 2D arrays and labels are a list of integers.
2. Set `aug=True` in train script.
3. Optionally update `aug_ratio` in config file to provide custom clean:aug images per batch(paper main results reported using 1:3).

## 📈 Main Results:
| dataset       | clean accuracy | VRA@36/255 | Lipschitz Constant | Autoattack@36/255 |
|:-------------:|:--------------:|:----------:|:------------------:|:-----------------:|
| CIFAR-10      | 71.77%         | 63.15%     |       192.0        |        65.04      |
| CIFAR-100     | 43.30%         | 34.13%     |       47.7         |        37.00      |
| Tiny-ImageNet | 36.15%         | 28.11%     |       26.69        |        32.08      |


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations
If you find this repository useful, consider to use the following citation:

```
@INPROCEEDINGS{
menon2025lipshift,
title={LipShi{FT}: A Certifiably Robust Shift-based Vision Transformer},
author={Rohan Menon and Nicola Franco and Stephan G{\"u}nnemann},
booktitle={ICLR 2025 Workshop: VerifAI: AI Verification in the Wild},
year={2025},
url={https://openreview.net/forum?id=OfxNKIHfUA}
}
```
