# TensorFlow2 Implementation of Denoising Diffusion Probabilistic Model


This is a TensorFlow2 implementation (tested on version 2.13.0) of **Denoising Diffusion Probabilistic Model** ([paper](https://arxiv.org/abs/2006.11239) and [official TF1 implementation](https://github.com/hojonathanho/diffusion)). You can also generate samples with [DDIM](https://arxiv.org/abs/2010.02502) (Denoising Diffusion Implicity Model)

## Quick Start

Clone this repo
```
git clone git@github.com:chao-ji/ddpm\_tf2.git
```

### Sampling

#### Prepare Checkpoint files

1. Download official checkpoint files
Download TF1 checkpoint files from [this link](https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja)

2. Convert to TF2-compatible formats
Run `python convert_to_tf2_ckpt.py` to convert them to TF2-compatible formats

#### Sample with DDPM 

Run

```
python sample.py --config_path config.yaml --model_path model_path
``` 

e.g. `python sample.py --config_path cifar10.yaml --model_path cifar10-1` to generate samples.

set `--store_prog` to `True` to save intermediate results


#### Sample with DDIM

Set `--use_ddim` to `True` to sample with DDIM

You can get pretty decent results with default parameters (`eta` being 0 and `ddim_steps` being 50). Or you can try larger `eta` values up to 1.0, and `ddim_steps` up to 1000 (This is when DDIM fall backs to DDPM).  

Set `--interpolate` to `True` to generate images using latents that are evenly interpolated between two independent latent noises

### Training
Run
```
python train.py --config_path config.yaml --ckpt_path dir_to_checkpoint
```
for training your own DDPM model.

### Samples of generated images

<p align="center">
  <img src="assets/celeb.png" width="1200">

  <br>
  Progressive sampling of CelebaHQ images
</p>


<p align="center">
  <img src="assets/bedroom.png" width="1200">

  <br>
  Progressive sampling of LSUN bedroom images
</p>


<p align="center">
  <img src="assets/church.png" width="1200">

  <br>
  Progressive sampling of LSUN church images
</p>


<p align="center">
  <img src="assets/cat.png" width="1200">

  <br>
  Progressive sampling of LSUN cat images
</p>

Note: for each series of progressive samples (4 in each dataset), the first row is the progressively denoised image, while the second row is the predicted original image.


<p align="center">
  <img src="assets/interpolate.png" width="1200">

  <br>
  Samples generated from interpolated latents using DDIM.
</p>
