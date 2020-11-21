import tensorflow as tf
from tensorflow.keras import layers

class SpectralNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer):
        super().__init__(layer)

        self._initialised = False
        self.w = self.u = self.w_shape = None

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalise()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def normalise(self):

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        v = tf.math.l2_normalize(tf.linalg.matmul(u, w, transpose_b=True))
        u = tf.math.l2_normalize(tf.linalg.matmul(v, w))

        sigma = tf.linalg.matmul(tf.matmul(v, w), u, transpose_b=True)

        self.w.assign(self.w / sigma)
        self.u.assign(u)


class SelfAttention1D(tf.keras.Model):
    def __init__(self, filters, spectral_norm=True):
        super().__init__(name='')

        if spectral_norm:
            self.f = SpectralNorm(layers.Conv1D(filters//8, 1, 1, 'same'))
            self.g = SpectralNorm(layers.Conv1D(filters//8, 1, 1, 'same'))
            self.h = SpectralNorm(layers.Conv1D(filters, 1, 1, 'same'))
        else:
            self.f = layers.Conv1D(filters//8, 1, 1, 'same')
            self.g = layers.Conv1D(filters//8, 1, 1, 'same')
            self.h = layers.Conv1D(filters, 1, 1, 'same')

        self.gamma = self.add_weight(initializer=tf.initializers.Zeros)

    def call(self, x):
        f_x = self.f(x)
        g_x = self.g(x)
        h_x = self.h(x)

        attn = tf.linalg.matmul(f_x, g_x, transpose_b=True)
        attn = tf.nn.softmax(attn)

        attn = self.gamma * tf.linalg.matmul(attn, h_x)

        return x + attn