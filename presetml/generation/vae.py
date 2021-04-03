""" VAE Implementation adapted from

https://blog.keras.io/building-autoencoders-in-keras.html
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

An archive of this article is saved in the reference directory. The full,
original script should be in the commit history.

"""
import uuid
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_core.python.keras.losses import MSE, binary_crossentropy
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from presetml import constants

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
from presetml.parsing import ableton_analog


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def run_vae(args, data):
    (x_train, _), (x_test, y_test) = data
    (
        batch_size,
        decoder,
        encoder,
        epochs,
        inputs,
        original_dim,
        outputs,
        vae,
        x_test,
        x_train,
        y_test,
        z_log_var,
        z_mean,
    ) = build_vae(x_train, x_test, y_test)

    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = MSE(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")
    vae.summary()
    plot_model(vae, to_file="vae_mlp.png", show_shapes=True)
    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
        )
        # vae.save_weights('vae_analog_presets.h5')
        n = 10
        grid_x = np.linspace(-1, 1, n)
        grid_y = np.linspace(-1, 1, n)[::-1]
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                for preset in x_decoded:
                    xml = ableton_analog.generate_xml_from_vector(x_decoded[0])
                    ableton_analog.write_zipped_preset(
                        f"{constants.ANALOG_PRESETS_PATH}/VAE/"
                        f"zsample-({yi},{xi})-{uuid.uuid4()}.adv",
                        xml,
                    )


def build_vae(x_train, x_test, y_test):
    preset_vector_size = x_train.shape[1]
    original_dim = preset_vector_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # network parameters
    input_shape = (original_dim,)
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 50
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name="encoder_input")
    x = Dense(intermediate_dim, activation="relu")(inputs)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    plot_model(encoder, to_file="vae_mlp_encoder.png", show_shapes=True)
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(intermediate_dim, activation="relu")(latent_inputs)
    outputs = Dense(original_dim, activation="sigmoid")(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name="decoder")
    decoder.summary()
    plot_model(decoder, to_file="vae_mlp_decoder.png", show_shapes=True)
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name="vae_mlp")
    return (
        batch_size,
        decoder,
        encoder,
        epochs,
        inputs,
        original_dim,
        outputs,
        vae,
        x_test,
        x_train,
        y_test,
        z_log_var,
        z_mean,
    )
