# READ
This repository is the official implementation of our ACL'23 Findings paper [Robust Natural Language Understanding with Residual Attention Debiasing](https://arxiv.org/abs/2305.17627).

## Installation
### Dependency
Experiments are run in the following environment:

| Package        | Version   |
|----------------|-----------|
| conda          |   22.9.0  |
| Python         |   3.8     |
| CUDA           |   11.8    |

### Install via Conda and Pip
```bash
conda create -n read python=3.8
conda activate read
pip install -r requirements.txt
```

## Data
Download:
- FEVER train/dev/symmetric test data from [FeverSymmetric GitHub repo](https://github.com/TalSchuster/FeverSymmetric)
- For PAWS-QQP:
  - Generate PAWS-QQP data by following the instructions in the README file of the [PAWS: Paraphrase Adversaries from Word Scrambling repo](https://github.com/google-research-datasets/paws)
  - Put the generated train.tsv and dev_and_test.tsv under [qqp_paws](dataset/qqp_paws) directory
The dataset directory should contain the following files:
```
└── dataset 
    └── fever
        ├── fever.dev.jsonl
        ├── fever.train.jsonl
        └── fever_symmetric_generated.jsonl
    └── qqp_paws
        ├── dev_and_test.tsv
        └── train.tsv
```

## Training
Parameters are defined in [train_ensemble.sh](train_ensemble.sh) script. Change the value of ```task_name``` to the desired task name (mnli, fever, qqp).  To train the ensemble model from scratch, please run the following
```
bash train_ensemble.sh
```

## Evaluation
Due to different number of labels that MNLI and HANS datasets have, please:
- Go to [eval_hans.sh](eval_hans.sh)
- Change value of ```model_name_or_path``` to your checkpoint
- Run the following to evaluate model on HANS dataset:
  ```
  bash eval_hans.sh
  ```

## Citation
```
@inproceedings{wang-etal-2023-robust,
    title = "Robust Natural Language Understanding with Residual Attention Debiasing",
    author = "Wang, Fei  and  Huang, James Y.  and  Yan, Tianyi  and  Zhou, Wenxuan  and  Chen, Muhao",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.32",
    doi = "10.18653/v1/2023.findings-acl.32",
    pages = "504--519",
}
```
