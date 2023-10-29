"""Train DDPM model."""
import yaml

import tensorflow as tf
from absl import app
from absl import flags

from dataset import create_cifar10_dataset, create_celebeahq256_dataset 
from model import UNet
from model_runner import DDPMTrainer 
from utils import LearningRateSchedule

flags.DEFINE_string("config_path", None, "Path to yaml config file.")
flags.DEFINE_string("ckpt_path", None, "Path to checkpoint directory.")
flags.DEFINE_integer("save_freq", 1000, "Frequency to save checkpoint.")
flags.DEFINE_string("dataset", "celebahq256", "Training dataset.")

FLAGS = flags.FLAGS


def main(_):
  with open(FLAGS.config_path) as f:
    config = yaml.safe_load(f)

  image_channels = config["image_channels"]
  image_size = config["image_size"]
  num_channels = config["num_channels"]

  channel_multipliers = config["channel_multipliers"]

  attention_resolutions = config["attention_resolutions"]

  beta_start = config["beta_start"]
  beta_end = config["beta_end"]

  num_steps = config["num_steps"]
  batch_size = config["batch_size"]
  epochs = config["epochs"]
  grad_clip = config["grad_clip"]

  learning_rate = config["learning_rate"]
  warmup = config["warmup"]
  dropout = config["dropout"]
  var_type = config["model_var_type"]

  ema_decay = config["ema_decay"]

  flip = config["flip"]



  schedule = LearningRateSchedule(learning_rate, warmup)

  adam = tf.keras.optimizers.Adam(
    learning_rate=schedule,
    use_ema=True,
    ema_momentum=ema_decay,  
    clipnorm=grad_clip,
  )

  unet = UNet(
      image_channels=image_channels,
      num_channels=num_channels,
      multipliers=channel_multipliers,
      attention_resolutions=attention_resolutions,
  )

  trainer = DDPMTrainer(
      unet, beta_start, beta_end, num_steps, model_var_type=var_type)

  if FLAGS.dataset == "cifar10":
    dataset = create_cifar10_dataset(
        batch_size=batch_size, epochs=epochs, flip=flip)
  else:
    dir_paths = config["dir_paths"]
    dataset = create_celebeahq256_dataset(
        dir_paths=dir_paths,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        flip=flip
    )

  ckpt = tf.train.Checkpoint(model=unet, optimizer=adam)
  ckpt_path = FLAGS.ckpt_path
  persist_per_iterations = FLAGS.save_freq

  trainer.train(
    dataset,
    adam,
    ckpt,
    ckpt_path,
    persist_per_iterations,
  )


if __name__ == "__main__":
  flags.mark_flag_as_required("config_path")
  flags.mark_flag_as_required("ckpt_path")
  app.run(main)
