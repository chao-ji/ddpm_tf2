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


def slerp(z1, z2, alpha):
  theta = tf.acos(tf.reduce_sum(z1 * z2) / (tf.norm(z1) * tf.norm(z2)))
  return (
      tf.sin((1 - alpha) * theta) / tf.sin(theta) * z1
      + tf.sin(alpha * theta) / tf.sin(theta) * z2
  )


class _DDPM(object):
  """Base model to be subclassed by DDPM trainer, sampler, and DDIM sampler."""
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
      model_var_type (str): type of variances (`sigma^{2}_t`). "fixed_large"
        refers to `beta_t`, and `fixed_small` refers to scaled `beta_t` that is
        smaller. See Section 3.2 of DDPM paper.
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


class DDPMSampler(_DDPM):
  """DDPM sampler."""
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
    """Sample a less noisy version of `xt` (i.e. x_{t-1})for one time step.

    Args:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      clip_denoised (bool): whether to clip the model output.
      return_pred_x0 (bool): whether to return pred_x0.

    Returns:
      sample (Tensor): tensor of shape [batch_size, height, width, 3], a
        less noisy version of `xt`.
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
      x_final (Tensor): tensor of shape [batch_size, height, width, 3], the
        final version of sampled image.
    """
    # decrement `t` from num_steps - 1 to 0, inclusively
    t = tf.constant(self._num_steps - 1, dtype="int32")
    xt = tf.random.normal(shape)
   
    x_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda t, _: tf.greater_equal(t, 0),
          body=lambda t, xt: [
              t - 1,
              self.p_sample(
                  xt=xt, t=tf.fill([shape[0]], t), return_pred_x0=False,
              )
          ],
          loop_vars=[t, xt],
          shape_invariants=[t.shape, xt.shape],
        ),
    )[1]
    return x_final

  def p_sample_loop_progressive(self, shape, record_freq=50):
    """Same as `p_sample_loop`, except that we record the intermediate results
    of sampling.

    Args:
      shape (tuple): shape of the image (i.e. [batch_size, height, width, 3])
      record_freq (int): the frequence with which to store the intermediate
        results.

    Returns:
      x_final (Tensor): tensor of shape [batch_size, height, width, 3], the
        final version of sampled image.
      sample_prog_final (Tensor): tensor of shape [batch_size, num_records,
        height, width, 3], the records of intermediate samples (sampled
        `x_{t-1}` from `xt` and `t`).
      pred_x0_prog_final (Tensor): tensor of shape [batch_size, num_records,
        height, width, 3], the records of predicted `x0`.
    """
    t = tf.constant(self._num_steps - 1, dtype="int32")
    xt = tf.random.normal(shape)

    num_records = self._num_steps // record_freq
    sample_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")
    pred_x0_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")

    def _loop_body(t, xt, sample_progress, pred_x0_progress):
      sample, pred_x0 = self.p_sample(
        xt=xt, t=tf.fill([shape[0]], t), return_pred_x0=True)
      insert_mask = tf.equal(tf.math.floordiv(t, record_freq),
                             tf.range(num_records, dtype="int32"))
      insert_mask = tf.reshape(
          tf.cast(insert_mask, dtype="float32"),
          [1, num_records, *([1] * len(shape[1:]))])
      new_sample_progress = insert_mask * sample[:, tf.newaxis] + (
          1. - insert_mask) * sample_progress
      new_pred_x0_progress = insert_mask * pred_x0[:, tf.newaxis] + (
          1. - insert_mask) * pred_x0_progress

      return [t - 1, sample, new_sample_progress, new_pred_x0_progress]

    _, x_final, sample_prog_final, pred_x0_prog_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda t, _1, _2, _3: tf.greater_equal(t, 0),
          body=_loop_body,
          loop_vars=[t, xt, sample_progress, pred_x0_progress],
          shape_invariants=[t.shape, xt.shape] + [sample_progress.shape] * 2
        )
    )
    return x_final, sample_prog_final, pred_x0_prog_final


class DDPMTrainer(_DDPM):
  """DDPM trainer."""
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

  def compute_loss(self, x0):
    """Compute the mean squared error between noise and predicted noise as the
    loss.

    Args:
      x0 (Tensor): tensor of shape [batch_size, height, width, 3], the original
        image.

    Returns:
      loss (Tensor): scalar tensor.
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
            logdir="log",
    ):
    """Perform training iterations.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient
        updates.
      ckpt: a tf.train.Checkpoint instance, saves or load weights to/from
        checkpoint file.
      ckpt_path: string scalar, the path to the directory that the checkpoint
        files will be written to or loaded from.
      persist_per_iterations: int scalar, saves weights to checkpoint files
        every `persist_per_iterations` iterations.
      log_per_iterations: int scalar, prints log info every `log_per_iterations`
        iterations.
      logdir: string scalar, the directory that the tensorboard log data will
        be written to.
    """
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
      print("Restoring from checkpoint: %s ..." % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print("Training from scratch...")

    for data in dataset:
      loss, step, lr = train_step(data)

      with summary_writer.as_default():
        tf.summary.scalar("train_loss", loss, step=step)
        tf.summary.scalar("learning_rate", lr, step=step)

      if step.numpy() % log_per_iterations == 0:
        print("global step: %d, loss: %f, learning rate:" %
            (step.numpy(), loss.numpy()), lr.numpy())
        sys.stdout.flush()
      if step.numpy() % persist_per_iterations == 0:
        print("Saving checkpoint at global step %d ..." % step.numpy())
        ckpt.save(os.path.join(ckpt_path, "ddpm"))

    optimizer.finalize_variable_values(self._model.trainable_variables)
    ckpt.save(os.path.join(ckpt_path, "ddpm"))



class DDIMSampler(_DDPM):
  """DDIM sample.

  DDIM falls back to DDPM when num_ddim_steps := num_steps and eta := 1.

  Ref: https://arxiv.org/abs/2010.02502
       https://github.com/ermongroup/ddim/tree/main
  """
  def __init__(
      self,
      model,
      num_ddim_steps=50,
      eta=0.,
      beta_start=0.0001,
      beta_end=0.02,
      num_steps=1000,
      model_var_type="fixed_small",
    ):
    """Constructor.

    Args:
      model (Layer): the UNet model.
      num_ddim_steps (int): number of DDIM steps.
      eta (float): eta controls the stochasticity of the reverse process.
      beta_start (float): start of beta schedule.
      beta_end (float): end of beta schedule.
      num_steps (int): the number of diffusion steps (`T`).
      model_var_type (str): type of variances (`sigma^{2}_t`). "fixed_large"
        refers to `beta_t`, and `fixed_small` refers to scaled `beta_t` that is
        smaller. See Section 3.2 of DDPM paper.
    """    
    super(DDIMSampler, self).__init__(
        model,
        beta_start=beta_start,
        beta_end=beta_end,
        num_steps=num_steps,
        model_var_type=model_var_type
    )
    self._num_ddim_steps = num_ddim_steps
    self._eta = eta

    self._ddim_steps = tf.range(
        0, num_steps, num_steps // num_ddim_steps, dtype="int32"
    )
    self._ddim_steps_prev = tf.concat([[-1], self._ddim_steps[:-1]], axis=0)
    self._alphas_cumprod = tf.math.cumprod(
        1 - tf.concat([[0.], self._betas], axis=0), axis=0
    )

  def ddim_sample(self, xt, index, return_pred_x0=False):
    """
    Args:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
      index (Tensor): index of the time step into `ddim_steps` of the
        subsequence used in DDIM. 
      return_pred_x0 (bool): whether to return pred_x0.

    Returns:
      sample (Tensor): tensor of shape [batch_size, height, width, 3], a
        less noisy version of `xt`.
      pred_x0 (Tensor): tensor of shape [batch_size, height, width, 3], the
        predicted `x0` in the process.
    """
    t = tf.fill([xt.shape[0]], self._ddim_steps[index])
    t_prev = tf.fill([xt.shape[0]], self._ddim_steps_prev[index])


    eps = self._model(xt, tf.cast(t, "float32"))
    alphas_cumprod = _extract(self._alphas_cumprod, t + 1)
    alphas_cumprod_prev = _extract(self._alphas_cumprod, t_prev + 1)

    pred_x0 = (xt - eps * tf.sqrt(1 - alphas_cumprod)) / tf.sqrt(alphas_cumprod)
    sigma = self._eta * tf.sqrt(
        (1 - alphas_cumprod / alphas_cumprod_prev) *
        (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    )
    noise = tf.random.normal(xt.shape)
    sample = tf.sqrt(alphas_cumprod_prev) * pred_x0 + tf.sqrt(
        1 - alphas_cumprod_prev - sigma ** 2) * eps + sigma * noise

    if return_pred_x0:
      return sample, pred_x0
    else:
      return sample

  def ddim_interpolate(self, shape, n=10):
    """Run reverse process on latents evenly (using slerp) interpolated between
    two latents `z1` and `z2`.

    Args:
      shape (list): shape of the latent variable.
      n (int): number of interpolation intervals.
 
    Returns:
      x_final (Tensor): the final version of sampled images for interpolated
        latents.
    """
    z1 = tf.random.normal(shape=[1] + shape)
    z2 = tf.random.normal(shape=[1] + shape)
    alphas = tf.cast(tf.range(n + 1), "float32") * 0.1
    index = tf.size(alphas) - 1

    z_inters = tf.zeros([0] + list(shape))
    z_inters = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            cond=lambda index, _,: tf.greater_equal(index, 0),
            body=lambda index, z_inters: (
                index - 1,
                tf.concat([slerp(z1, z2, alphas[index]), z_inters], axis=0)
            ),
            loop_vars=[index, z_inters],
            shape_invariants=[[None] + index.shape[1:], z_inters]
        ),
    )[1]

    x_final = self.ddim_p_sample_loop(xt=z_inters)
    return x_final

  def ddim_p_sample_loop(self, shape=None, xt=None):
    """Sampling loop to generate samples according to batched image shape.

    Args:
      shape (tuple): shape of the image (i.e. [batch_size, height, width, 3])
      xt (Tensor): latents of shape [batch_size, height, width, 3]

    note: either `shape` or `xt` must be provided.
    
    Returns:
      x_final (Tensor): tensor of shape [batch_size, height, width, 3], the
        final version of sampled image.
    """
    assert not (shape is None and xt is None)
    index = tf.size(self._ddim_steps) - 1
    if shape is not None:
      xt = tf.random.normal(shape)

    x_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda index, _: tf.greater_equal(index, 0),
          body=lambda index, xt: [
              index - 1,
              self.ddim_sample(
                  xt=xt, index=index, return_pred_x0=False,
              )
          ],
          loop_vars=[index, xt],
          shape_invariants=[index.shape, xt.shape],
        ),
    )[1]
    return x_final

  def ddim_p_sample_loop_progressive(self, shape, record_freq=5):
    """Same as `p_sample_loop`, except that we record the intermediate results
    of sampling.

    Args:
      shape (tuple): shape of the image (i.e. [batch_size, height, width, 3])
      record_freq (int): the frequence with which to store the intermediate
        results.

    Returns:
      x_final (Tensor): tensor of shape [batch_size, height, width, 3], the
        final version of sampled image.
      sample_prog_final (Tensor): tensor of shape [batch_size, num_records,
        height, width, 3], the records of intermediate samples (sampled
        `x_{t-1}` from `xt` and `t`).
      pred_x0_prog_final (Tensor): tensor of shape [batch_size, num_records,
        height, width, 3], the records of predicted `x0`.
    """
    index = tf.size(self._ddim_steps) - 1
    xt = tf.random.normal(shape)

    num_records = tf.size(self._ddim_steps) // record_freq
    sample_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")
    pred_x0_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")

    def _loop_body(index, xt, sample_progress, pred_x0_progress):
      sample, pred_x0 = self.ddim_sample(
        xt=xt, index=index, return_pred_x0=True)
      insert_mask = tf.equal(tf.math.floordiv(index, record_freq),
                             tf.range(num_records, dtype="int32"))
      insert_mask = tf.reshape(
          tf.cast(insert_mask, dtype="float32"),
          [1, num_records, *([1] * len(shape[1:]))])
      new_sample_progress = insert_mask * sample[:, tf.newaxis] + (
          1. - insert_mask) * sample_progress
      new_pred_x0_progress = insert_mask * pred_x0[:, tf.newaxis] + (
          1. - insert_mask) * pred_x0_progress

      return [index - 1, sample, new_sample_progress, new_pred_x0_progress]

    _, x_final, sample_prog_final, pred_x0_prog_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda index, _1, _2, _3: tf.greater_equal(index, 0),
          body=_loop_body,
          loop_vars=[index, xt, sample_progress, pred_x0_progress],
          shape_invariants=[index.shape, xt.shape] + [sample_progress.shape] * 2
        )
    )
    return x_final, sample_prog_final, pred_x0_prog_final
