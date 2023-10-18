import math
import tensorflow as tf

GROUP_NORM_EPSILON = 1e-6

 
class TimeEmbedding(tf.keras.layers.Layer):
  """Time embedding layer. This layer absorbs the two dense layers immediately
  after the `get_timestep_embedding` function in the original implementation.
  """
  def __init__(self, num_channels, activation=tf.nn.swish):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
      activation (callable): the activation function.
    """
    super(TimeEmbedding, self).__init__()
    self._num_channels = num_channels
    self._activation = activation

    self._dense1 = tf.keras.layers.Dense(num_channels)
    self._dense2 = tf.keras.layers.Dense(num_channels)

  def call(self, time):
    """Computes the time embeddings.

    Args:
      time (Tensor): tensor of shape [batch_size], the time step indieces.

    Returns:
      outputs (Tensor): tensor of shape [batch_size, num_channels], the time
        embeddings in a batch.
    """
    num_channels = self._num_channels // 8
    outputs = math.log(10000) / (num_channels - 1)
    outputs = tf.exp(tf.range(num_channels, dtype="float32") * -outputs)
    outputs = tf.cast(time, "float32")[:, tf.newaxis] * outputs[tf.newaxis]
    outputs = tf.concat([tf.sin(outputs), tf.cos(outputs)], axis=1)
    if self._num_channels % 2 == 1:
      outputs = tf.pad(outputs, [[0, 0], [0, 1]])

    outputs = self._activation(self._dense1(outputs))
    outputs = self._dense2(outputs)

    batch_size = time.shape[0]
    assert outputs.shape == [batch_size, self._num_channels]
    return outputs


class ResidualBlock(tf.keras.layers.Layer):
  """Residual block that combines time embeddings with the images feature maps.
  """
  def __init__(
      self,
      num_channels,
      num_groups=32,
      dropout_rate=0.1,
      activation=tf.nn.swish,
    ):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
      num_groups (int): num of groups in group normalization layer.
      dropout_rate (float): dropout rate.
      activation (callable or str): activation function.
    """
    super(ResidualBlock, self).__init__()

    self._num_channels = num_channels
    self._num_groups = num_groups
    self._dropout_rate = dropout_rate
    self._activation = activation

    self._group_norm1 = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON,
    )
    self._conv1 = tf.keras.layers.Conv2D(
        num_channels, kernel_size=3, padding="same",
    )
    self._group_norm2 = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON,
    )
    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._conv2 = tf.keras.layers.Conv2D(
        num_channels, kernel_size=3, padding="same",
    )
    self._shortcut = tf.keras.layers.Dense(num_channels)
    self._dense_time = tf.keras.layers.Dense(num_channels)

  def call(self, inputs, time, training=False):
    """Compute the output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      time (Tensor): time embedding tensor of shape [batch_size,
        num_time_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, height, width,
        outputs].
    """
    outputs = self._conv1(self._activation(self._group_norm1(inputs)))

    # add time embedding
    outputs += self._dense_time(self._activation(time))[:, None, None]
    outputs = self._conv2(self._dropout(
        self._activation(self._group_norm2(outputs)), training=training
    ))

    if inputs.shape[-1] != self._num_channels:
      inputs = self._shortcut(inputs)

    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs


class AttentionBlock(tf.keras.layers.Layer):
  """Attention block that computes output feature map by computing
  self-attention within input feature map.
  """
  def __init__(self, num_channels, num_groups=32):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
      num_groups (int): num of groups in group normalization layer.
    """
    super(AttentionBlock, self).__init__()
    self._num_channels = num_channels
    self._num_groups = num_groups

    self._group_norm = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON,
    )
    self._dense_query = tf.keras.layers.Dense(num_channels)
    self._dense_key = tf.keras.layers.Dense(num_channels)
    self._dense_value = tf.keras.layers.Dense(num_channels)
    self._dense_output = tf.keras.layers.Dense(num_channels)

  def call(self, inputs):
    """Compute the output tensor with attention mechanism.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, height, width,
        num_channels].
    """
    batch_size, height, width = inputs.shape[:3]

    outputs = self._group_norm(inputs)

    # [batch_size, height, width, num_channels]
    q = self._dense_query(outputs)
    # [batch_size, height, width, num_channels]
    k = self._dense_key(outputs)
    # [batch_size, height, width, num_channels]
    v = self._dense_value(outputs)

    # [batch_size, height, width, height, width]
    attention_weights = tf.einsum(
        "bhwc,bHWc->bhwHW", q, k) * self._num_channels ** -0.5
    attention_weights = tf.reshape(
        attention_weights, [batch_size, height, width, height * width]
    )
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    attention_weights = tf.reshape(
        attention_weights, [batch_size, height, width, height, width]
    )
    outputs = tf.einsum("bhwHW,bHWc->bhwc", attention_weights, v)
    outputs = self._dense_output(outputs) 
    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs


class DownBlock(tf.keras.layers.Layer):
  """Down block that wraps a `ResidualBlock` and `AttentionBlock` in the
  downsampling pathway.

  `AttentionBlock` is used ONLY when the feature map size is in the tuple
  `attention_resolutions`.
  """
  def __init__(self, num_channels, attention_resolutions):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
      attention_resolutions (tuple): tuple of feature map size.
    """
    super(DownBlock, self).__init__()
    self._num_channels = num_channels
    self._attention_resolutions = attention_resolutions

    self._res = ResidualBlock(num_channels)
    self._attention = AttentionBlock(num_channels)

  def call(self, inputs, time, training=False):
    """Compute output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      time (Tensor): time embedding tensor of shape [batch_size,
        num_time_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, height, width,
        num_channels].
    """
    outputs = self._res(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs 


class UpBlock(tf.keras.layers.Layer):
  """Up block that wraps a `ResidualBlock` and `AttentionBlock` in the
  upsampling pathway.

  `AttentionBlock` is used ONLY when the feature map size is in the tuple
  `attention_resolutions`.
  """
  def __init__(self, num_channels, attention_resolutions):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
      attention_resolutions (tuple): tuple of feature map size.
    """
    super(UpBlock, self).__init__()
    self._num_channels = num_channels
    self._attention_resolutions = attention_resolutions

    self._res = ResidualBlock(num_channels)
    self._attention = AttentionBlock(num_channels)

  def call(self, inputs, time, training=False):
    """Compute output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      time (Tensor): time embedding tensor of shape [batch_size,
        num_time_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, height, width,
        num_channels].
    """
    outputs = self._res(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs 


class MiddleBlock(tf.keras.layers.Layer):
  """Middle block that wraps a `ResidualBlock`, `AttentionBlock` and another
  `ResidualBlock`.
  """
  def __init__(self, num_channels):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
    """
    super(MiddleBlock, self).__init__()
    self._res1 = ResidualBlock(num_channels)
    self._attention = AttentionBlock(num_channels)
    self._res2 = ResidualBlock(num_channels)

  def call(self, inputs, time, training=False):
    """Compute output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      time (Tensor): time embedding tensor of shape [batch_size,
        num_time_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, height, width,
        num_channels].
    """
    outputs = self._res1(inputs, time)
    outputs = self._attention(outputs)
    outputs = self._res2(outputs, time)
    return outputs


class Upsample(tf.keras.layers.Layer):
  """Upsample feature map with a Resize followed by a stride-1 Conv2D."""
  def __init__(self, num_channels):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
    """
    super(Upsample, self).__init__()
    self._num_channels = num_channels

    self._conv = tf.keras.layers.Conv2D(
        num_channels, kernel_size=3, strides=1, padding="SAME"
    )

  def call(self, inputs, time=None, training=False):
    """Compute output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, out_height,
        out_width, num_channels].
    """
    _, height, width, _ = inputs.shape
    outputs = tf.raw_ops.ResizeNearestNeighbor(
        images=inputs, size=[height * 2, width *2], align_corners=True
    )
    outputs = self._conv(outputs)
    return outputs


class Downsample(tf.keras.layers.Layer):
  """Downsample feature map with a stride-2 Conv2D."""
  def __init__(self, num_channels):
    """Constructor.

    Args:
      num_channels (int): num of output channels.
    """
    super(Downsample, self).__init__()
    self._num_channels = num_channels

    self._conv = tf.keras.layers.Conv2D(
        num_channels, kernel_size=3, strides=2, padding="SAME"
    )

  def call(self, inputs, time=None, training=False):
    """Compute output tensor.

    Args:
      inputs (Tensor): input tensor of shape [batch_size, height, width,
        num_channels].
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): output tensor of shape [batch_size, out_height,
        out_width, num_channels].
    """
    outputs = self._conv(inputs)
    return outputs


class UNet(tf.keras.layers.Layer):
  """Unet backbone."""
  def __init__(
      self,
      image_channels=3,
      num_channels=128,
      multipliers=(1, 2, 2, 2),
      attention_resolutions=(16,),
      num_blocks=2,
      activation=tf.nn.swish,
    ):
    """Constructor.

    Args:
      image_channels (int): num of input image channels.
      num_channels (int): num of base output channels for each `DownBlock` or
        `UpBlock`.
      multipliers (tuple): the output channels for each `DownBlock` or
        `UpBlock` will be `num_channels` times values in `multipliers`.
      attention_resolutions (tuple): feature maps whose `height` or `width` size
        are in this tuple will go though `AttentionBlock`.
      num_blocks (int): num of times each `DownBlock` or `UpBlock` will be
        repeated.
      activation (callable or str): activation function.
    """
    super(UNet, self).__init__()

    self._image_channels = image_channels
    self._num_channels = num_channels
    self._multipliers = multipliers
    self._attention_resolutions = attention_resolutions
    self._num_blocks = num_blocks
    self._activation = activation

    num_resolutions = len(multipliers)
    channels_list = [num_channels * mul for mul in multipliers]

    self._conv_in = tf.keras.layers.Conv2D(
        num_channels, kernel_size=3, padding="SAME")
    self._position_encoding = TimeEmbedding(num_channels * 4)

    down = []
    for i in range(num_resolutions):
      for j in range(num_blocks):
        down.append(DownBlock(channels_list[i], attention_resolutions))
      if i < num_resolutions - 1:
        down.append(Downsample(channels_list[i]))

    self._down = down
    self._middle = MiddleBlock(channels_list[-1])

    up = []
    for i in reversed(range(num_resolutions)):
      for j in range(num_blocks + 1):
        up.append(UpBlock(channels_list[i], attention_resolutions))
      if i > 0:
        up.append(Upsample(channels_list[i]))
    self._up = up

    self._group_norm = tf.keras.layers.GroupNormalization(
        32, epsilon=GROUP_NORM_EPSILON)
    self._conv_out = tf.keras.layers.Conv2D(
        image_channels, kernel_size=3, padding="SAME")

  def call(self, inputs, time, training=False):
    """Compute predicted noise (epsilon).

    Args:
      inputs (Tensor): input image of shape [batch_size, image_height,
        image_width, num_image_channels].
      time (Tensor): tensor of shape [batch_size], the time step indieces.
      training (bool): True if in training mode.

    Returns:
      outputs (Tensor): tensor of shape [batch_size, image_height, image_width,
        num_image_channels].
    """
    time = self._position_encoding(time)
    outputs = self._conv_in(inputs)
    hiddens = [outputs]

    for m in self._down:
      outputs = m(outputs, time, training=training)
      hiddens.append(outputs)

    outputs = self._middle(outputs, time)

    for m in self._up:
      if isinstance(m, Upsample):
        outputs = m(outputs, time, training=training)
      else:
        s = hiddens.pop()
        outputs = tf.concat([outputs, s], axis=-1)
        outputs = m(outputs, time, training=training)

    outputs = self._conv_out(self._activation(self._group_norm(outputs)))
    return outputs


if __name__ == "__main__":
  import numpy as np


  """
  unet = UNet(attention_resolutions=(16,))
  ckpt = tf.train.Checkpoint(model=unet)
  ckpt_reader = tf.train.load_checkpoint("diffusion_cifar10_model/model.ckpt-790000")
 
  weights = [] 
  with open("diffusion_cifar10_model/kv") as f:
    for line in f:
      line = line.strip()
      if not len(line):
        continue
      weights.append(ckpt_reader.get_tensor(line.split(" ")[0]))

  inputs = tf.constant(np.random.uniform(-1, 1, (50, 32, 32, 3)).astype("float32"))
  time = tf.constant(np.arange(50).astype("float32"))
  unet(inputs, time, training=False)
  unet.set_weights(weights)
  out = unet(inputs, time, training=False) 

  ckpt.save("cifar10")

  #"""


  #"""
  model = UNet(attention_resolutions=(16,), multipliers=(1, 1, 2, 2, 4, 4))
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_reader = tf.train.load_checkpoint("diffusion_lsun_bedroom_model/model.ckpt-2388000")

  weights = []
  with open("diffusion_celeba_hq_model/kv") as f:
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

  ckpt.save("lsun_bedroom")
  #"""
  

