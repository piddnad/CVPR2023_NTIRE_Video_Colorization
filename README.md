

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/modelscope)](https://pypi.org/project/modelscope/)
<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/modelscope)](https://GitHub.com/modelscope/modelscope/commit/)
[![Leaderboard](https://img.shields.io/badge/ModelScope-Check%20Your%20Contribution-orange)](https://opensource.alibaba.com/contribution_leaderboard/details?projectValue=modelscope)

<!-- [![GitHub contributors](https://img.shields.io/github/contributors/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/graphs/contributors/) -->
<!-- [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) -->


</div>

# NTIRE 2023 Video Colorization Model and Dataset

This project provides the baseline model and evaluation code for track1 and track2 for CVPR 2023 NTIRE workshop Video Colorization Challenge.

## Installation

```
conda create -n video_colorization python=3.7
conda activate video_colorization

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

git clone https://github.com/piddnad/CVPR2023_NTIRE_Video_Colorization.git

cd CVPR2023_NTIRE_Video_Colorization
pip install -r requirements/tests.txt
pip install -r requirements/framework.txt
pip install -r requirements/cv.txt

```


## Download Dataset (Optional)

You can Run the code below to download the validation set:

```
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode


# Set dataset download path
cache_dir = './datasets' 

# Download validation set
val_set = MsDataset.load('ntire23_video_colorization', namespace='damo', subset_name='val_frames', split='validation', cache_dir=cache_dir, download_mode=DownloadMode.FORCE_REDOWNLOAD)
print(next(iter(val_set)))
```


## Baseline Evaluation on Validation Set

This step will automatically download the validation set.

```
cd CVPR2023_NTIRE_Video_Colorization
CUDA_VISIBLE_DEVICES=0  PYTHONPATH=. python ntire23_scripts/baseline_evaluation.py

# Then you might get output similar to:
# FID evaluation time: xxxx
# CDC evaluation time: xxxx
# Total evaluation time: xxxx
# FID: 47.15574537543114, CDC: 0.003475072230336491

```


## Evaluation on Your Results

First modify the `res_dir` in user_result_evaluation.py, and then run:

```
python ntire23_scripts/user_result_evaluation.py
```

# About ModelScope

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

[ModelScope]( https://www.modelscope.cn) is built upon the notion of “Model-as-a-Service” (MaaS). It seeks to bring together most advanced machine learning models from the AI community, and streamlines the process of leveraging AI models in real-world applications. The core ModelScope library open-sourced in this repository provides the interfaces and implementations that allow developers to perform  model inference, training and evaluation.


In particular, with rich layers of API-abstraction, the ModelScope library offers unified experience to explore state-of-the-art models spanning across domains such as CV, NLP, Speech, Multi-Modality, and Scientific-computation. Model contributors of different areas can integrate models into the ModelScope ecosystem through the layered-APIs, allowing easy and unified access to their models. Once integrated, model inference, fine-tuning, and evaluations can be done with only a few lines of codes. In the meantime, flexibilities are also provided so that different components in the model applications can be customized wherever necessary.

Apart from harboring implementations of a wide range of different models, ModelScope library also enables the necessary interactions with ModelScope backend services, particularly with the Model-Hub and Dataset-Hub. Such interactions facilitate management of  various entities (models and datasets) to be performed seamlessly under-the-hood, including entity lookup, version control, cache management, and many others.

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
