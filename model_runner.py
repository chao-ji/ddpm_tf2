"""Denosing Diffusion Probabilisitc Model for training and sampling.

Ref: https://github.com/hojonathanho/diffusion/diffusion_tf/diffusion_utils_2.py
"""
import sys
import os

import tensorflow as tf
import numpy as np


def _extract(data, t):
  """Extract some coefficients at specified time steps, then reshape to
  [batch_size, 1, 1, 1] for broadcasting purpose.

  Args:
    data (Tensor): tensor of shape [num_steps], coefficients for a beta
      schedule.
    t (Tensor): int tensor of shape [batch_size], sampled time steps in a batch.

  Returns:
    outputs (Tensor): tensor of shape [batch_size, 1, 1, 1], the extracted
      coefficients.
  """
  outputs = tf.reshape(
      tf.gather(tf.cast(data, dtype="float32"), t, axis=0),
      [-1, 1, 1, 1]
  )
  return outputs


class DenoiseDiffusion(object):
  """DDPM model."""
  def __init__(
      self,
      model,
      beta_start=0.0001,
      beta_end=0.02,
      num_steps=1000,
      model_var_type="fixed_small",
    ):
    """Constructor.

    Args:
      model (Layer): the UNet model.
      beta_start (float): start of beta schedule.
      beta_end (float): end of beta schedule.
      num_steps (int): the number of diffusion steps (`T`).
    """
    self._model = model
    self._beta_start = beta_start
    self._beta_end = beta_end
    self._num_steps = num_steps
    self._model_var_type = model_var_type

    # linear beta schedule; set dtype to float64 for better precision
    self._betas = tf.cast(
        tf.linspace(beta_start, beta_end, num_steps), "float64")

    # pre-compute all coefficients used in a beta schedule
    self._alphas = 1. - self._betas
    self._alphas_cumprod = tf.math.cumprod(self._alphas, axis=0)
    self._alphas_cumprod_prev = tf.concat(
        [[1.], self._alphas_cumprod[:-1]], axis=0)
    self._sqrt_alphas_cumprod = tf.sqrt(self._alphas_cumprod)
    self._sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self._alphas_cumprod)
    self._log_one_minus_alphas_cumprod = tf.math.log(1. - self._alphas_cumprod)
    self._sqrt_recip_alphas_cumprod = tf.sqrt(1. / self._alphas_cumprod)
    self._sqrt_recipm1_alphas_cumprod = tf.sqrt(1. / self._alphas_cumprod - 1)
    self._posterior_variance = self._betas * (
        1. - self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)
    self._posterior_log_variance_clipped = tf.math.log(
        tf.concat([[self._posterior_variance[1]],
                    self._posterior_variance[1:]], axis=0))
    self._posterior_mean_coef1 = self._betas * tf.sqrt(
        self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)
    self._posterior_mean_coef2 = (1. - self._alphas_cumprod_prev) * tf.sqrt(
        self._alphas) / (1. - self._alphas_cumprod)
    self._log_betas_clipped = tf.math.log(
        tf.concat([[self._posterior_variance[1]], self._betas[1:]], axis=0))

  def q_sample(self, x0, t, eps):
    """Sample a noised version of `x0` (original image) according to `t` (
    sampled time steps), i.e. `q(x_t | x_0)`. `t` contains integers sampled from
    0, 1, ..., num_steps - 1, so 0 means 1 diffusion step, and so on.

    Args:
      x0 (Tensor): tensor of shape [batch_size, height, width, 3], the original
        image.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      eps (Tensor): tensor of shape [batch_size, height, width, 3], noise from
        prior distribution.

    Returns:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
    """
    xt = (
        _extract(self._sqrt_alphas_cumprod, t) * x0 +
        _extract(self._sqrt_one_minus_alphas_cumprod, t) * eps 
    )
    return xt

  def _predict_x0_from_eps(self, xt, t, eps):
    """Predict `x0` from epsilon by rearranging the following equation:

    xt = alphas_cumprod_t**0.5*x0 + (1-alphas_cumprod_t)**0.5*epsilon
    (Eq. 4 of DDPM paper)

    Args:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      eps (Tensor): tensor of shape [batch_size, height, width, 3], noise from
        prior distribution.

    Returns:
      pred_x0 (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted `x0`.
    """
    pred_x0 = (
        _extract(self._sqrt_recip_alphas_cumprod, t) * xt -
        _extract(self._sqrt_recipm1_alphas_cumprod, t) * eps
    )
    return pred_x0

  def q_posterior_mean_variance(self, x0, xt, t):
    """Compute the mean and log variance of the diffusion posterior
    `q(x_{t-1} | x_t, x_0) = coef1 * x_0 + coef2 * x_t` (Eq. 7 of DDPM paper)

    Args:
      x0 (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted `x0`.
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.

    Returns:
      posterior_mean (Tensor): tensor of shape [batch_size, height, width, 3],
        mean of the diffusion posterior.
      posterior_log_variance_clipped (Tensor): tensor of shape [batch_size, 1,
        1, 1], log variance of the diffusion posterior.
    """
    posterior_mean = (
        _extract(self._posterior_mean_coef1, t) * x0 +
        _extract(self._posterior_mean_coef2, t) * xt
    )
    posterior_log_variance_clipped = _extract(
        self._posterior_log_variance_clipped, t)
    return posterior_mean, posterior_log_variance_clipped

  def p_mean_variance(self, xt, t, clip_denoised):
    """Compute the mean and log variance of the reverse distribution
    `p(x_{t-1} | x_t)` (Eq. 1 of DDPM paper), where the mean is predicted (eps
    prediction mode), and the variance is fixed.

    Args:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      clip_denoised (bool): whether to clip the model output.

    Returns:
      model_mean (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted mean of the reverse distribution.
      model_log_variance (Tensor): tensor of shape [batch_size, height, width,
        3], log variance of the reverse distribution.
      pred_x0 (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted `x0` in the process.
    """
    eps = self._model(xt, tf.cast(t, "float32"))

    if self._model_var_type == "fixed_small":
      model_log_variance = _extract(self._posterior_log_variance_clipped, t
          ) * tf.ones(xt.shape.as_list())
    elif self._model_var_type == "fixed_large":
      model_log_variance = _extract(self._log_betas_clipped, t
          ) * tf.ones(xt.shape.as_list())
    else:
      raise NotImplementedError(f"Unsupported var type: {self._model_var_type}")

    _maybe_clip = lambda x: (tf.clip_by_value(x, -1, 1) if clip_denoised else x)
    # eps prediction mode
    pred_x0 = _maybe_clip(self._predict_x0_from_eps(xt=xt, t=t, eps=eps))
    model_mean, _ = self.q_posterior_mean_variance(x0=pred_x0, xt=xt, t=t)
    return model_mean, model_log_variance, pred_x0

  def p_sample(self, xt, t, clip_denoised=True, return_pred_x0=False):
    """Sample a denoised version for one time step.

    Args:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      clip_denoised (bool): whether to clip the model output.
      return_pred_x0 (bool): whether to return pred_x0.

    Returns:
      sample (Tensor): tensor of shape [batch_size, height, width, 3], a
        *less noisy* version of `xt`.
      pred_x0 (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted `x0` in the process.
    """
    model_mean, model_log_variance, pred_x0 = self.p_mean_variance(
      xt=xt, t=t, clip_denoised=clip_denoised)
    noise = tf.random.normal(xt.shape)

    # mask out noise when t == 0 (no noise for last step of the reverse process)
    mask = tf.reshape(
        1 - tf.cast(tf.equal(t, 0), "float32"),
        [xt.shape[0]] + [1] * (len(xt.shape) - 1)
    )
    sample = model_mean + mask * tf.exp(0.5 * model_log_variance) * noise
    if return_pred_x0:
      return sample, pred_x0
    else:
      return sample

  def p_sample_loop(self, shape):
    """Sampling loop to generate samples according to batched image shape.

    Args:
      shape (tuple): shape of the image (i.e. [batch_size, height, width, 3])

    Returns:
      x_final (Tensor)
    """
    # decrement `t` from num_steps - 1 to 0, inclusively
    t = tf.constant(self._num_steps - 1, dtype="int32")
    xt = tf.random.normal(shape)
   
    x_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda i, _: tf.greater_equal(i, 0),
          body=lambda i, img: [
              i - 1,
              self.p_sample(
                  xt=img, t=tf.fill([shape[0]], i), return_pred_x0=False,
              )
          ],
          loop_vars=[t, xt],
          shape_invariants=[t.shape, xt.shape],
        ),
    )[1]
    return x_final

  def p_sample_loop_progressive(self, shape, include_pred_x0_freq=50):
    t = tf.constant(self._num_steps - 1, dtype="int32")
    xt = tf.random.normal(shape)

    num_recorded_pred_x0 = self._num_steps // include_pred_x0_freq
    progress = tf.zeros(
        [shape[0], num_recorded_pred_x0, *shape[1:]], dtype="float32")

    def _loop_body(i, img, progress):
      sample, pred_x0 = self.p_sample(
        xt=img, t=tf.fill([shape[0]], i), return_pred_x0=True)
      insert_mask = tf.equal(tf.math.floordiv(i, include_pred_x0_freq),
                             tf.range(num_recorded_pred_x0, dtype="int32"))
      insert_mask = tf.reshape(
          tf.cast(insert_mask, dtype="float32"),
          [1, num_recorded_pred_x0, *([1] * len(shape[1:]))])
      new_progress = insert_mask * pred_x0[:, tf.newaxis] + (
          1. - insert_mask) * progress
      return [i - 1, sample, new_progress]

    _, img_final, xstartpreds_final = tf.nest.map_structure(
      tf.stop_gradient,
      tf.while_loop(
      cond=lambda i, img, progress: tf.greater_equal(i, 0),
      body=_loop_body,
      loop_vars=[t, xt, progress],
      shape_invariants=[t.shape, xt.shape, progress.shape],
    ))
    return img_final, xstartpreds_final

  def compute_loss(self, x0):
    """Compute the mean squared error between noise and predicted noise as the
    loss.

    Args:
      x0 (Tensor): 

    Returns:
      loss (Tensor):
    """
    batch_size = tf.shape(x0)[0]
    t = tf.random.uniform((batch_size,), 0, self._num_steps, dtype="int32")
    noise = tf.random.normal(tf.shape(x0))
    xt = self.q_sample(x0, t, noise)
    eps = self._model(xt, tf.cast(t, "float32"), training=True)
    loss = tf.reduce_mean(tf.math.squared_difference(noise, eps), axis=[1, 2, 3])
    loss = tf.reduce_mean(loss)
    return loss

  def train(self,
            dataset,
            optimizer,
            ckpt,
            ckpt_path,
            persist_per_iterations,
            log_per_iterations=100,
            logdir='log'
    ):
    train_step_signature = [
        tf.TensorSpec(
            shape=dataset.element_spec.shape,
            dtype=dataset.element_spec.dtype,
        ),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(images):
      with tf.GradientTape() as tape:
        loss = self.compute_loss(images) 

      gradients = tape.gradient(loss, self._model.trainable_variables)
      optimizer.apply_gradients(
          zip(gradients, self._model.trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate
      return loss, step - 1, lr

    summary_writer = tf.summary.create_file_writer(logdir)

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print('Training from scratch...')

    for data in dataset: 
      loss, step, lr = train_step(data)

      with summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=step)
        tf.summary.scalar('learning_rate', lr, step=step)

      if step.numpy() % log_per_iterations == 0:
        print('global step: %d, loss: %f, learning rate:' %
            (step.numpy(), loss.numpy()), lr.numpy())
        sys.stdout.flush()
      if step.numpy() % persist_per_iterations == 0:
        print('Saving checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_path, 'ddpm'))

    optimizer.finalize_variable_values(self._model.trainable_variables)
    ckpt.save(os.path.join(ckpt_path, "ddpm")) 

