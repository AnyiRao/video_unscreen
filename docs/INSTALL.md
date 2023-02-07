## Installation and Preparation

![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.0.0-Orange?logo=pytorch) ![mmcv](https://img.shields.io/badge/mmcv-%3E%3D0.4.0-green)


### Requirements
SceneSeg is currently very easy to install before the introduction of **feature extractor**

- Python 3.6+
- PyTorch 1.0 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

a. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```sh
conda install pytorch torchvision -c pytorch
```

b. Clone the video_unscreen repository.

```sh
git clone https://github.com/AnyiRao/video_unscreen.git
cd video_unscreen
```
c. Install Python Packages

```sh
pip install -r docs/requirements.txt
```

### Folder Structure
```sh
|-data ## the data_root for experiments
|-pre  ## for preprocess
|-video_unscreen
|   |-unscreen
|   |-configs
|   |-weights
|   |-tools
```

### Download Pretrained Models and Data
Download model weights and data [here](https://drive.google.com/drive/folders/1IYcUaimgllu_PyE6jBHDgn8FfD4stdkW?usp=sharing)
and put them in the `data` and `weights` folder.
