import tensorflow as tf
from tensorflow import keras
from larq.layers import QuantDense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training import distribution_strategy_context


def log2(x):
    """Computes the base 2 logarithm on an input"""
    tf.math.log(x) / tf.math.log(2.0)


@tf.custom_gradient
def AP2(x):
    """Returns the approximate power of 2 of the input."""
    return 2 ** (tf.round(log2(tf.abs(x)))), lambda dy: dy


@tf.custom_gradient
def FixedPointQuantize(inputs, scale, bits, rescale):
    """Apply fixed point quantization to an input tensor."""
    # Clip values into specified range
    y = tf.clip_by_value(inputs, -scale, scale)
    # Determine floating point value for each bit
    bit_value = scale / (2.0 ** bits - 1.0)
    # Quantize tensor
    y = y / bit_value
    y = tf.round(y)
    # Readjust to floating point if specified
    y = tf.cond(rescale, true_fn=lambda: y * bit_value, false_fn=lambda: y)

    def grad_fn(dy):
        grad_mask = tf.cast(tf.abs(inputs) <= scale, tf.float32)
        dx = grad_mask * dy
        return [dx, None, None, None]

    return y, grad_fn


def get_quantize_bits(x):
    """Computes approximate mean and 1-bit value for an input weight tensor."""
    if len(x.shape) > 2:
        mean = tf.reduce_mean(tf.abs(tf.reshape(x, [-1, x.shape[-1]])), axis=0)
    else:
        mean = tf.reduce_mean(tf.abs(x))

    # Fix dimensions of mean if needed
    for i in range(len(x.shape) - 1):
        mean = tf.expand_dims(mean, axis=0)
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2 * bits) - 1
    approximate_mean = AP2(mean)
    return approximate_mean, bits


def compute_quantized_shiftnorm(
    variance, mean, epsilon, latent_weights, extra_scale, bits, rescale=True
):
    """Computes approximated shiftnorm deviation and mean."""
    # Compute number of bits to shift for std division.
    std_factor = 1.0 / (extra_scale * tf.sqrt(variance + epsilon))
    approximate_std = AP2(std_factor)
    # Now determine total number of bits needed for fixed point quantization of mean.
    weight_scale_ap2, _ = get_quantize_bits(latent_weights)
    weight_scale_bits = -log2(weight_scale_ap2)
    weight_scale_bits = tf.reshape(weight_scale_bits, [-1])
    total_shift_bits = weight_scale_bits + bits

    # Determine quantization scale based on geometric series.
    mean_scale = 1.0 + ((1.0 / (2.0 ** bits - 1.0)) * (1.0 - (1.0 / 2.0 ** weight_scale_bits)))

    # Now quantize each channel of mean appropriately
    quantized_means = FixedPointQuantize(mean, mean_scale, total_shift_bits, rescale)

    return approximate_std, quantized_means


def get_shiftnorm_ap2(layer, latent_weights, rescale=False):
    """Helper function for extracting std and mean from shiftnorm layer."""
    mean = layer.weights[0].value()
    extra_scale = layer.extra_scale
    epsilon = layer.epsilon
    variance = layer.weights[1].value()
    bits = layer.bits
    approximate_std, quantized_means = compute_quantized_shiftnorm(
        variance, mean, epsilon, latent_weights, extra_scale, bits, rescale
    )
    return approximate_std, quantized_means


def BatchNormalization(use_shiftnorm, *args, **kwargs):
    """Helper function that returns either a shiftnorm or batchnorm layer."""
    if use_shiftnorm:
        return ShiftNormalization(*args, **kwargs)
    else:
        return keras.layers.BatchNormalization(*args, **kwargs)


class ShiftNormalization(keras.layers.BatchNormalization):
    """Shift normalization layer

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Arguments:
    previous_layer: The keras layer that precedes this one.
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    binary_dense: Set true when using after a binary dense layer.
        Need some special handling for batchnorm for binary dense layers to
        prevent small variance.
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

    def __init__(self, bits, previous_layer, **kwargs):
        super(ShiftNormalization, self).__init__(**kwargs)
        self.bits = bits
        self.previous_layer = previous_layer
        self.binary_dense = isinstance(previous_layer, QuantDense)
        self.extra_scale = self.scope.shiftnorm_scale

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError("Input has undefined rank:", input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError("Invalid axis: %d" % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError("Duplicate axis: %s" % self.axis)

        if self.virtual_batch_size is not None:
            if self.virtual_batch_size <= 0:
                raise ValueError(
                    "virtual_batch_size must be a positive integer that "
                    "divides the true batch size of the input Tensor"
                )
            # If using virtual batches, the first dimension must be the batch
            # dimension and cannot be the batch norm axis
        if 0 in self.axis:
            raise ValueError(
                "When using virtual_batch_size, the batch dimension "
                "must be 0 and thus axis cannot include 0"
            )
        if self.adjustment is not None:
            raise ValueError("When using virtual_batch_size, adjustment cannot " "be specified")

        if self.fused in (None, True):
            # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
            # output back to its original shape accordingly.
            if self._USE_V2_BEHAVIOR:
                if self.fused is None:
                    self.fused = ndims == 4
            elif self.fused and ndims != 4:
                raise ValueError(
                    "Batch normalization layers with fused=True only " "support 4D input tensors."
                )
        else:
            assert self.fused is not None
            self.fused = ndims == 4 and self._fused_can_be_used()
        # TODO(chrisying): fused batch norm is currently not supported for
        # multi-axis batch norm and by extension virtual batches. In some cases,
        # it might be possible to use fused batch norm but would require reshaping
        # the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
        # particularly tricky. A compromise might be to just support the most
        # common use case (turning 5D w/ virtual batch to NCHW)

        if self.fused:
            if self.axis == [1]:
                self._data_format = "NCHW"
            elif self.axis == [3]:
                self._data_format = "NHWC"
            else:
                raise ValueError(
                    "Unsupported axis, fused batch norm only supports " "axis == [1] or axis == [3]"
                )

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError("Input has undefined `axis` dimension. Input shape: ", input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if self.binary_dense:
            param_shape = [1]
        elif len(axis_to_dim) == 1 and self.virtual_batch_size is None:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)]
            if self.virtual_batch_size is not None:
                # When using virtual batches, add an extra dim at index 1
                param_shape.insert(1, 1)
                for idx, x in enumerate(self.axis):
                    self.axis[idx] = x + 1  # Account for added dimension

        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None
            if self.fused:
                self._gamma_const = K.constant(1.0, dtype=self._param_dtype, shape=param_shape)

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None
            if self.fused:
                self._beta_const = K.constant(0.0, dtype=self._param_dtype, shape=param_shape)

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, "_scope") and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None

            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            if self.renorm:
                # In batch renormalization we track the inference moving stddev instead
                # of the moving variance to more closely align with the paper.
                def moving_stddev_initializer(*args, **kwargs):
                    return tf.math.sqrt(self.moving_variance_initializer(*args, **kwargs))

                with distribution_strategy_context.get_strategy().extended.colocate_vars_with(
                    self.moving_variance
                ):
                    self.moving_stddev = self.add_weight(
                        name="moving_stddev",
                        shape=param_shape,
                        dtype=self._param_dtype,
                        initializer=moving_stddev_initializer,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf.VariableAggregation.MEAN,
                        experimental_autocast=False,
                    )

                # Create variables to maintain the moving mean and standard deviation.
                # These are used in training and thus are different from the moving
                # averages above. The renorm variables are colocated with moving_mean
                # and moving_stddev.
                # NOTE: below, the outer `with device` block causes the current device
                # stack to be cleared. The nested ones use a `lambda` to set the desired
                # device and ignore any devices that may be set by the custom getter.
                def _renorm_variable(name, shape, initializer=tf.zeros_initializer()):
                    """Create a renorm variable."""
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        dtype=self._param_dtype,
                        initializer=initializer,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf.VariableAggregation.MEAN,
                        experimental_autocast=False,
                    )
                    return var

                with distribution_strategy_context.get_strategy().extended.colocate_vars_with(
                    self.moving_mean
                ):
                    self.renorm_mean = _renorm_variable(
                        "renorm_mean", param_shape, self.moving_mean_initializer
                    )
                with distribution_strategy_context.get_strategy().extended.colocate_vars_with(
                    self.moving_stddev
                ):
                    self.renorm_stddev = _renorm_variable(
                        "renorm_stddev", param_shape, moving_stddev_initializer
                    )
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def call(self, inputs, training=None):
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

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        # For dense layers, require a full reduction
        if self.binary_dense:
            reduction_axes = [i for i in range(ndims)]
        # Otherwise, reduce all but the feature axis
        else:
            reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1)):
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

            # When norming the output of a binary dense layer, make sure the shape is maintained.
            if self.binary_dense:
                mean = tf.reshape(mean, [1])
                variance = tf.reshape(variance, [1])

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
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
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
                    true_branch = lambda: _do_update(self.moving_variance, new_variance)

                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)

        # Extract weights of preceding layer to calculate means.
        previous_weights = self.previous_layer.weights[0].value()
        approximate_std, quantized_means = compute_quantized_shiftnorm(
            variance,
            mean,
            self.epsilon,
            previous_weights,
            self.extra_scale,
            self.bits,
            rescale=True,
        )

        outputs = inputs - quantized_means
        outputs = outputs * approximate_std

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs
