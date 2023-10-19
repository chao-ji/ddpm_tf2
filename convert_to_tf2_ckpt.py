import tensorflow as tf
import numpy as np

from model import UNet


def convert_small_model(ckpt_path, keys_filename, ckpt_name="cifar10"):
  unet = UNet(attention_resolutions=(16,))
  ckpt = tf.train.Checkpoint(model=unet)
  ckpt_reader = tf.train.load_checkpoint(ckpt_path)
 
  weights = [] 
  with open(keys_filename) as f:
    for line in f:
      line = line.strip()
      if not len(line):
        continue
      weights.append(ckpt_reader.get_tensor(line.split(" ")[0]))

  inputs = tf.constant(np.random.uniform(-1, 1, (2, 32, 32, 3)).astype("float32"))
  time = tf.constant(np.arange(2).astype("float32"))
  unet(inputs, time, training=False)
  unet.set_weights(weights)
  out = unet(inputs, time, training=False) 
  ckpt.save(ckpt_name)


def convert_large_model(ckpt_path, keys_filename, ckpt_name="celebahq256"):
  model = UNet(attention_resolutions=(16,), multipliers=(1, 1, 2, 2, 4, 4))
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_reader = tf.train.load_checkpoint(ckpt_path)

  weights = []
  with open(keys_filename) as f:
    for line in f:
      line = line.strip()
      if not len(line):
        continue
      weights.append(ckpt_reader.get_tensor(line.split(" ")[0]))

  inputs = tf.constant(np.random.uniform(-1, 1, (2, 256, 256, 3)).astype("float32"))
  time = tf.constant(np.arange(2).astype("float32"))
  model(inputs, time, training=False)
  model.set_weights(weights)
  out = model(inputs, time, training=False)
  ckpt.save(ckpt_name)


if __name__ == "__main__":
  convert_small_model("diffusion_cifar10_model/model.ckpt-790000", "small_model_keys")
  print("finished converting cifar10 model")
  convert_large_model("diffusion_lsun_bedroom_model/model.ckpt-2388000", "large_model_keys", "lsun_bedroom")
  print("finished converting lsun bedroom model")
  convert_large_model("diffusion_lsun_cat_model/model.ckpt-1761000", "large_model_keys", "lsun_cat") 
  print("finished converting lsun cat model")
  convert_large_model("diffusion_lsun_church_model/model.ckpt-4432000", "large_model_keys", "lsun_church")
  print("finished converting lsun church model")
  convert_large_model("diffusion_celeba_hq_model/model.ckpt-560000", "large_model_keys", "celebahq256")
  print("finished converting celebahq256 model")
