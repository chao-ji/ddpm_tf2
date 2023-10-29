import tensorflow as tf


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Warmup learning rate schedule."""
  def __init__(self, learning_rate, warmup_steps):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step.

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= tf.minimum(global_step / self._warmup_steps, 1.0)

    return learning_rate
