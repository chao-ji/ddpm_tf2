"""Sample images using pretrained diffusion models."""
import yaml

import tensorflow as tf

from absl import app
from absl import flags

import numpy as np
from model import UNet
from model_runner import DDPMSampler, DDIMSampler


flags.DEFINE_string("model_path", None, "Path to trained DDPM modeli.")
flags.DEFINE_string("config_path", None, "Path to yaml config file.")
flags.DEFINE_bool("use_ddim", False, "Whether to use DDIM (True) or DDPM"
    "(False) for sampling")
flags.DEFINE_float("eta", 0., "Eta parameter used in DDIM.")
flags.DEFINE_integer("ddim_steps", 50, "Number of DDIM steps.")
flags.DEFINE_bool("store_prog", True, "Whether to store progressive results.")
flags.DEFINE_bool("interpolate", False, "Wheter to sample from interpolated"
    " latents in DDIM.")
flags.DEFINE_integer("record_freq", 50, "Frequence with which to store"
    " intermediate results.")
flags.DEFINE_integer("num_samples", 4, "Number of images to be sampled.")
FLAGS = flags.FLAGS


def to_image(inputs):
  """Convert float type tensor of shape [B, H, W, C] to image."""
  for i in range(inputs.shape[0]):
    inputs[i] = (inputs[i] - inputs[i].min()) / (inputs[i].max() - inputs[i].min())

  outputs = (inputs * 255).astype("uint8")
  return outputs


def main(_):

  with open(FLAGS.config_path) as f: 
    config = yaml.safe_load(f)

  image_channels = config["image_channels"]
  image_size = config["image_size"]
  num_channels = config["num_channels"]
  channel_multipliers = config["channel_multipliers"]
  num_steps = config["num_steps"]
  var_type = config["model_var_type"]
  beta_start = config["beta_start"]
  beta_end = config["beta_end"]
  attention_resolutions = config["attention_resolutions"]

  unet = UNet(
      image_channels=image_channels,
      num_channels=num_channels,
      multipliers=channel_multipliers,
      attention_resolutions=attention_resolutions,
      )

  ckpt = tf.train.Checkpoint(model=unet)
  ckpt.restore(FLAGS.model_path)
  shape = [FLAGS.num_samples, image_size, image_size, 3]

  if FLAGS.use_ddim:
    sampler = DDIMSampler(
        unet,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        model_var_type=var_type,
        eta=FLAGS.eta,
        num_ddim_steps=FLAGS.ddim_steps,
    )

    if FLAGS.store_prog:
      images, sample_prog, pred_x0_prog = sampler.ddim_p_sample_loop_progressive(
          shape, record_freq=FLAGS.record_freq)
      np.save("sample_prog", to_image(sample_prog.numpy()))
      np.save("pred_x0_prog", to_image(pred_x0_prog.numpy()))
      print("Intermediate results saved to `sample_prog.npy` (x_{t-1}) and "
          "`pred_x0_png` (predicted x_0)")
    elif FLAGS.interpolate:
      images = sampler.ddim_interpolate(shape[1:])
    else:
      images = sampler.ddim_p_sample_loop(shape)

  else:
    sampler = DDPMSampler(
        unet,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        model_var_type=var_type,
    )

    if FLAGS.store_prog:
      images, sample_prog, pred_x0_prog = sampler.p_sample_loop_progressive(shape)
      np.save("sample_prog", to_image(sample_prog.numpy()))
      np.save("pred_x0_prog", to_image(pred_x0_prog.numpy()))
      print("Intermediate results saved to `sample_prog.npy` (x_{t-1}) and "
          "`pred_x0_png` (predicted x_0)")
    else:
      images = sampler.p_sample_loop(shape)

  np.save("images", to_image(images.numpy()))
  print("Sampled images saved to `images.npy`")


if __name__  == "__main__":
  flags.mark_flag_as_required("model_path")
  flags.mark_flag_as_required("config_path")
  app.run(main)

