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

Download `vgg_normalized.pth`(required for training) and `net_final.pth`(for testing only) and put them under `models/`

`vgg_normalized.pth`:

- Google Drive: https://drive.google.com/open?id=108uza-dsmwvbW2zv-G73jtVcMU_2Nb7Y

`net_final.pth`:

- Google Drive:https://drive.google.com/file/d/1e7ph-4cf8OKCp8nitXqHcOT0C9_otM1O



### 3. Test

Use `--content_dir`, `--texture_dir` and `--color_dir` to specify directories that save content images, texture reference images and color reference images. The model will iterate over all combinations between content, texture and color.

Use `--test_opt` to specify the type of test you want to conduct:

- `TC`: Transfer texture and color together.
- `ST`: Traditional style transfer.
- `T`: Texture only transfer.
- `C`: Color only transfer.
- `INT`: Parameter interpolation for texture and color. For this type of test, you need to specify the interpolation num by `--int_num`

```python
python main.py --mode test --test_opt <TEST_OPTION> --content_dir <CONTENT_DIR> --texture_dir <TEXTURE_DIR> --color_dir <COLOR_DIR>
```

For more detailed configurations, please refer to `--help` option. 



### 4. Train

Use `--content_dir`, `--texture_dir` and `--color_dir` to specify directories that save content images, texture reference images and color reference images.

In each iteration, the model will randomly sample a batch of content-texture-color pair for training. The training will stop when it reaches the maximum iteration num (specified by `--max_iter`). Usually the training will not iterate over the whole dataset as the num of combinations of content, texture and color is really large. 

```python
python main.py --mode train --content_dir <CONTENT_DIR> --texture_dir <TEXTURE_DIR> --color_dir <COLOR_DIR> 
```

For more detailed configurations, please refer to `--help` option. 



## Contact

If you have any questions or suggestions about this paper, feel free to contact me (z8hu@ucsd.edu).