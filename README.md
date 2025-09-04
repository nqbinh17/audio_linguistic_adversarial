# What You Read Isn’t What You Hear: Linguistic Sensitivity in Deepfake Speech Detection

[![arXiv](https://img.shields.io/badge/arXiv-2505.17513-b31b1b.svg)](https://arxiv.org/abs/2505.17513)

The official codebase of "What You Read Isn’t What You Hear" - EMNLP 2025 publication.

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Website](#website)
- [Citation](#citation)

# Audio Linguistic Adversarial Research

This repository contains code and resources for researching adversarial attacks in audio linguistics, specifically focusing on the linguistic sensitivity of deepfake speech detection.

## Introduction

This repository contains the code and resources for our paper, "What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection." It provides the framework for transcript-level adversarial attacks against audio anti-spoofing systems and the tools for comprehensive feature analysis.

## Setup Instructions

Executing the below script will guide you through the Miniconda installation and then proceed with the environment setup:

```bash
chmod +x setup.sh
bash setup.sh
```

## Usage

To activate your environment and run an attack:

```bash
conda activate ling_adv
python attack.py
```

## Website
Please visit our website for more information: https://lethaiq.github.io/linguistic-sensitivity-deepfake-voice/

## Citation

If you use this code or resources in your research, please cite our paper:

```bibtex
@misc{nguyen2025readisnthearlinguistic,
      title={What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection}, 
      author={Binh Nguyen and Shuji Shi and Ryan Ofman and Thai Le},
      year={2025},
      eprint={2505.17513},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17513}, 
}
```