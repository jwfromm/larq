import tensorflow as tf
from tensorflow import keras
import larq as lq
from larq import utils
from larq.layers import QuantDense
from larq.quantizers import QuantizerFunctionWrapper
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training import distribution_strategy_context


def log2(x):
    """Computes the base 2 logarithm on an input"""
    return tf.math.log(x) / tf.math.log(2.0)


@tf.custom_gradient
def smooth_round(x):
    return tf.round(x), lambda dy: dy


@tf.custom_gradient
def AP2(x):
    """Returns the approximate power of 2 of the input."""
    return 2 ** (tf.round(log2(tf.abs(x)))), lambda dy: dy


def _clipped_gradient(x, dy, clip_value, unipolar):
    if clip_value is None:
        return dy

    zeros = tf.zeros_like(dy)
    if not unipolar:
        x = tf.math.abs(x)
    mask = tf.math.less_equal(x, clip_value)
    if unipolar:
        mask = tf.math.logical_and(mask, tf.math.greater_equal(x, 0))
    return tf.where(mask, dy, zeros)


@utils.register_keras_custom_object
@utils.set_precision(1)
def linaer_quantize_ste(x: tf.Tensor, bits: int = 1, clip_value: float = 1.0, unipolar: bool = False) -> tf.Tensor:
    r"""N-Bit binarization using linear approximation.

    Supports both unipolar and bipolar quantization at various bitwidths.

    # Arguments
    x: Input tensor.
    clip_value: Threshold for clipping gradients. If 'None' gradients are not clipped.
    bits: Number of bits to quantize to.
    unipolar: If True, quantize in the range [0, 1]

    # Returns
    Binarized tensor.
    """
    bit_constant = (2**bits) - 1
    @tf.custom_gradient
    def _bipolar_call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value, False)

        # Force appropriate range
        qx = tf.clip_by_value(x, -1, 1)
        # Adjust to [0, 1] range
        qx = (qx + 1) / 2
        # Quantize in available bins.
        qx = tf.round(bit_constant * qx) / bit_constant
        # Readjust to [-1, 1]
        qx = 2 * (qx - 0.5)

        return qx, grad

    @tf.custom_gradient
    def _unipolar_call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value, True)

        # Force appropriate range
        qx = tf.clip_by_value(x, 0, 1)
        # Quantize in available bins
        qx = tf.round(bit_constant * qx) / bit_constant

        return qx, grad

    if unipolar:
        return _unipolar_call(x)
    return _bipolar_call(x)


@utils.register_keras_custom_object
class LinearQuantizer(QuantizerFunctionWrapper):
    def __init__(self, bits: int, clip_value: float = 1.0, unipolar: bool = False):
        super().__init__(linaer_quantize_ste, bits=bits, clip_value=clip_value, unipolar=unipolar)
        self.bits = bits
        self.clip_value = clip_value
        self.unipolar = unipolar


class BatchNormalization(keras.layers.BatchNormalization):
    """Shift normalization layer

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Arguments:
    latent_weights: Tensor, the weights of the previous layer.
    bits: Integer, How many bits activations are quantized with.
    unipolar: If true then unipolar quantization is being used, otherwise bipolar.
    use_shiftnorm: Bool, whether to use shiftnorm or regular batchnorm.
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
    trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    References:
        - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self, bits, unipolar=False, use_shiftnorm=True, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.bits = bits
        self.unipolar = unipolar 
        self.use_shiftnorm = use_shiftnorm
        self.fused = False

    def call(self, inputs, training=None):
        # Unpack inputs.
        if not self.use_shiftnorm:
            return super(BatchNormalization, self).call(inputs, training)

        training = self._get_training_value(training)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = tf.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = tf.reshape(outputs, original_shape)
                return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (
                v is not None
                and len(v.shape) != ndims
                and reduction_axes != list(range(ndims - 1))
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if (
            training_value == False
        ):  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(
                    training, lambda: adj_scale, lambda: tf.ones_like(adj_scale)
                )
                adj_bias = tf_utils.smart_cond(
                    training, lambda: adj_bias, lambda: tf.zeros_like(adj_bias)
                )
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                tf.cast(inputs, self._param_dtype), reduction_axes, keep_dims=keep_dims
            )

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(
                training, lambda: mean, lambda: tf.convert_to_tensor(moving_mean)
            )
            variance = tf_utils.smart_cond(
                training, lambda: variance, lambda: tf.convert_to_tensor(moving_variance)
            )

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = tf.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = tf.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                inputs_size = tf.size(inputs)
            else:
                inputs_size = None
            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training, inputs_size
                )
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(tf.stop_gradient(r, name="renorm_r"))
                d = _broadcast(tf.stop_gradient(d, name="renorm_d"))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum, inputs_size)

            def mean_update():
                def true_branch(): return _do_update(self.moving_mean, new_mean)
                def false_branch(): return self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(
                        self.moving_stddev, tf.sqrt(new_variance + self.epsilon)
                    )
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon),
                    )

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    def true_branch(): return _do_update(self.moving_variance, new_variance)

                def false_branch(): return self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)

        std = tf.sqrt(variance + self.epsilon)

        if scale is not None:
            std = std / scale

        if offset is not None:
            adjusted_offset = offset * std
            mean = mean - adjusted_offset

        # Quantize std and mean.
        std = 1 / std
        std = AP2(std)
        mean = (2**self.bits * smooth_round(mean)) / (2 ** self.bits)

        outputs = (inputs - _broadcast(mean)) * _broadcast(std)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs
