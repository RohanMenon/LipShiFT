# LipShiFT: A Certifiably Robust Shift-based Vision Transformer


This repository provides the implementation of our certifiably robust vision transformer. The code presented here is part of a paper accepted to the ICLR 2025 Workshop ([VerifAI](https://iclr.cc/virtual/2025/workshop/23973)).

The codebase is adapted from the work on LiResNet ([NeurIPS 2023](https://arxiv.org/abs/2301.12549)) and CHORD-LiResNet ([ICLR 2024](https://openreview.net/forum?id=qz3mcn99cu))


## Getting Started:

### Creating a development environment:
Install the latest Anaconda distribution into your local/remote session and use the following command to create a viable environment using the `environment.yaml` with all the dependencies required.
```conda env create -f environment.yml ```
Now activate the environment using the following command:
``` conda activate liresnet ``` 

### Training and Evaluation:
Now you can follow along the code and custom configs as described below:
1. For training a model, check out our `run.sh` as a starting point. 
2. Dive into our [configs](/configs) for additional dataset configurations.


Our training script used to generate results reported is: `train_lipshift.py`. And for evaluation of trained checkpoints, we use  `test_script.py`.

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

<!-- ## Citations
If you find this repository useful, consider to use the following citations

```
@INPROCEEDINGS{hu2023scaling,
    title={Unlocking Deterministic Robustness Certification on ImageNet},
    author={Kai Hu and Andy Zou and Zifan Wang and Klas Leino and Matt Fredrikson},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=SHyVaWGTO4}
}

@misc{hu2023recipe,
    title={A Recipe for Improved Certifiable Robustness: Capacity and Data}, 
    author={Kai Hu and Klas Leino and Zifan Wang and Matt Fredrikson},
    year={2023},
    eprint={2310.02513},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@INPROCEEDINGS{leino21gloro,
    title = {Globally-Robust Neural Networks},
    author = {Klas Leino and Zifan Wang and Matt Fredrikson},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021}
}
``` -->
