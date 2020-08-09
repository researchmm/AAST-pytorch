**We are currently refining our code, the final version will come soon!**



# AAST-pytorch

This is the pytorch implementation of **Aesthetic-Aware Image Style Transfer** [Hu et al., MM2020].



Our work propose a novel problem called Aesthetic-Aware Image Style Transfer(AAST) which aims to control the texture and color of an image independently during style transfer and thus generate results with more diverse aesthetic effects.



Our code framework refers to the framework of [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

## Results

- Results on Aesthetic Aware Image Style Transfer

  <img src="img/TC.pdf" alt="Aesthetic Aware Image Style Transfer" style="zoom:50%;" />

- Results on parameter Interpolation

  <img src="img/P-16.pdf" alt="Parameter Interpolation" style="zoom:50%;" />

## Usage

### 1.Prerequisite

- python 3.7+
- pytorch 1.4+
- torchvision 0.5+
- Pillow

(Optional)

- CUDA 10.0

- tqdm
- TensorboardX

### 2. Download Pre-trained model

Download `vgg_normalized.pth` and `net_final.pth` and put them under `models/`

`vgg_normalized.pth`:

- Google Drive: https://drive.google.com/open?id=108uza-dsmwvbW2zv-G73jtVcMU_2Nb7Y

`net_final.pth`:

- Google Drive:https://drive.google.com/file/d/1e7ph-4cf8OKCp8nitXqHcOT0C9_otM1O





## Contact

If you have any questions or suggestions about this paper, feel free to contact me (z8hu@ucsd.edu).