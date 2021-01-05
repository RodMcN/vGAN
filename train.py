import tensorflow as tf
from model import vGAN
import glob

def parse_image_function(example_proto):
    image_feature_description = {
        'data': tf.io.FixedLenFeature([16000], tf.float32),
        'label': tf.io.FixedLenFeature([35], tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)

BATCH_SIZE = 32
LR = 1e-4

def train(model, dataset, epochs=50, steps=20000):
    it = iter(dataset)

    if e < 25:
        if (e+1) % 5 == 0:
            steps = steps // 2
    
    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    t = time.time()

    ### TRAIN DISC ###
    for s in range(steps):

        real = next(it)
        real = real['data']

        if np.random.uniform() < 0.3:
            real = real / tf.reduce_max(real)
        elif np.random.uniform() < 0.4:
            real = real / abs(tf.reduce_min(real))
        if np.random.uniform() < 0.05:
            real = -real

        r = tf.random.uniform((1,), -2000, 4000, tf.int32)[0]
        real = tf.roll(real, r, 1)
        real = tf.expand_dims(real, -1)
        real = real[:, ::2, :]

        fake = tf.random.normal((batch_size, model.noise_dim, 1))
        fake = model.generator(fake)
         #augment with noise
        if np.random.uniform() < 0.3:
            scale = tf.random.uniform((1,), 0.001, 0.1)
            noise = tf.random.normal(fake.shape, 0, scale)
            fake2 = fake + noise
        else:
            fake2 = fake
        if np.random.uniform() < 0.3:
            scale = tf.random.uniform((1,), 0.001, 0.1)
            noise = tf.random.normal(real.shape, 0, scale)
            real2 = real + noise
        else:
            real2 = real

        disc_loss = model.disc_step(real2, fake2)
        disc_metric.update_state(disc_loss)

        t2 = time.time() - t
        m = int(t2 // 60)
        s = int(t2 % 60)

        print(f"\r{e+1}/{epochs}, generator: {gen_metric.result():.4f}, discriminator: {disc_metric.result():.4f}, {m}m{s}s", end="")
    
    for s in range(steps):

        gen_loss = model.gen_step(batch_size)
        gen_metric.update_state(gen_loss)
        
        t2 = time.time() - t
        m = int(t2 // 60)
        s = int(t2 % 60)

        print(f"\r{e+1}/{epochs}, generator: {gen_metric.result():.4f}, discriminator: {disc_metric.result():.4f}, {m}m{s}s", end="")
    print()



if __name__ == "__main__":
    tfr = glob.glob("./datasets/*.tfr")
    ds = tf.data.TFRecordDataset(tfr, 'GZIP')

    dataset = ds.map(parse_image_function)\
        .shuffle(2500)\
        .batch(BATCH_SIZE, drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)\
        .repeat()
    
    model = vGAN(BATCH_SIZE, attention=False)


    model.gen_opt = tf.keras.optimizers.RMSprop(LR)
    model.disc_opt = tf.keras.optimizers.RMSprop(LR)

    model.make_gen()
    model.make_disc(dropout=0.7, attn=False)

    train(model, dataset)

    
