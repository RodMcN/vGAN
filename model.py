import tensorflow as tf
from tensorflow.keras import layers, initializers
import math
from custom_layers import SpectralNorm, SelfAttention1D
import time

class vGAN():
    def __init__(self, batch_size, wgan=True, attention=True, normalisation='spectral', transpose_upsampling=False, opt='adam'):
        self.wgan = wgan
        self.attn = attention
        assert normalisation in ['spectral', 'batch', 'none', None]
        self.normalise = normalisation
        self.trans_up = transpose_upsampling

        self.noise_dim = 125

        self.generator = None
        self.discriminator = None

        self.batch_size = batch_size

        self.gp_weight = 10

        if opt == 'sgd':
            self.gen_opt = tf.keras.optimizers.SGD(1e-5, momentum=0.9,)
            self.disc_opt = tf.keras.optimizers.SGD(1e-5, momentum=0.9,)
        elif opt == 'adam':
            self.gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9,)
            self.disc_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9,)

    def add_conv_block(self, model, filters, stride, upscale):

        bias = self.normalise in ['none', None]

        if upscale:
            if self.trans_up:
                l = layers.Conv1DTranspose(filters, 3, stride, 'same', kernel_initializer=initializers.he_normal, use_bias=bias)
            else:
                model.add(layers.UpSampling1D(size=stride))
                l = layers.Conv1D(filters, 3, 1, 'same', kernel_initializer=initializers.he_normal, use_bias=bias)
        else:
            l = layers.Conv1D(filters, 3, stride, 'same', kernel_initializer=initializers.he_normal, use_bias=bias)

        if self.normalise == 'spectral':
            model.add(SpectralNorm(l))
        elif self.normalise == 'batch':
            model.add(l)
            model.add(layers.BatchNormalization())
        else:
            model.add(l)
        model.add(layers.LeakyReLU())


    def make_gen(self, target_dim=8000):
        upscales = math.log((target_dim/self.noise_dim), 2)
        assert upscales % 1 == 0

        model = tf.keras.Sequential()

        filters = 128
        for i in range(int(upscales)):
            self.add_conv_block(model, filters, 2, True)
            if self.attn and (i == int(upscales * 0.3) or i == int(upscales * 0.75)):
                model.add(SelfAttention1D(filters))
                model.add(layers.LeakyReLU())
            filters = max(32, filters//2)

        model.add(layers.Conv1D(1, 3, 1, 'same', kernel_initializer=initializers.he_normal))
        if not self.wgan:
            model.add(layers.Activation(tf.keras.activations.tanh))

        self.generator = model

    def make_disc(self, filters=None, dropout=0.0, attn=None):
        model = tf.keras.Sequential()

        if filters is None:
            filters = [32, 32, 64, 64, 128]

        if attn is None:
            attn = self.attn

        for i, f in enumerate(filters):
            if attn and (i == int(len(filters) * 0.3) or i == int(len(filters) * 0.75)):
                model.add(SelfAttention1D(f))
                model.add(layers.LeakyReLU())

            stride = 2 if i < 3 else 1
            self.add_conv_block(model, f, stride, False)
            if dropout > 0:
                model.add(layers.Dropout(dropout))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        if not self.wgan:
            model.add(layers.Activation(tf.keras.activations.sigmoid))

        self.discriminator = model

    def gradient_penalty(self, real_inp, fake_inp):
        r = tf.random.uniform([self.batch_size, 1, 1], 0, 1)
        interpolated = (r * real_inp) + ((1-r) * fake_inp)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return penalty

    def disc_loss(self, real_preds, fake_preds, smoothing=0):
        if self.wgan:
            real_loss = tf.reduce_mean(real_preds)
            fake_loss = tf.reduce_mean(fake_preds)

            return fake_loss - real_loss

        else:
            if smoothing > 0:
                smoothing = tf.random.uniform((1,), 0, smoothing)
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_preds)-smoothing, real_preds)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(real_preds)+smoothing, real_preds)

            return real_loss + fake_loss

    def gen_loss(self, fake_preds):
        if self.wgan:
            return -tf.reduce_mean(fake_preds)
        else:
            return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_preds), fake_preds)

    def disc_step(self, real_inp, fake_inp):
        with tf.GradientTape() as tape:
            real_preds = self.discriminator(real_inp, training=True)
            fake_preds = self.discriminator(fake_inp, training=True)
            loss = self.disc_loss(real_preds, fake_preds)

            penalty = self.gradient_penalty(real_inp, fake_inp)

            loss = loss + (self.gp_weight * penalty)

        grads = tape.gradient(loss, self.discriminator.trainable_variables)

        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        return loss

    def gen_step(self, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim, 1])

        with tf.GradientTape() as tape:
            generated = self.generator(noise, training=True)
            preds = self.discriminator(generated, training=False)
            loss = self.gen_loss(preds)
        grads = tape.gradient(loss, self.generator.trainable_variables)

        self.disc_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return loss
