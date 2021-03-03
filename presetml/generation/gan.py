""" Example of training a gan on mnist

Taken from:
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

This page is saved in the reference directory
"""
import os
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import GaussianNoise

from presetml.generation.util import wasserstein_loss, ClipConstraint
from presetml.parsing import ableton_analog
from presetml import constants
from presetml import datatypes
from matplotlib import pyplot

verbose = True

from tensorflow.keras.utils import plot_model


def run(dataset: np.array, dest_preset_path: str, config: datatypes.GANConfig):
    vectors, labels = dataset
    # create the gan
    d_model, g_model, gan_model = build_gan(data_len=vectors.shape[1],
                                            config=config)
    # train model
    train(g_model, d_model, gan_model, vectors, labels,
          dest_preset_path, config)


def build_gan(data_len, config: datatypes.GANConfig):
    d_model = define_critic(in_dim=data_len, parameters=config)
    g_model = define_generator(out_dim=data_len,
                              parameters=config)
    gan_model = define_gan(g_model, d_model, config=config)
    return d_model, g_model, gan_model


# define the standalone discriminator model
def define_critic(in_dim: int = 216,
                 parameters: datatypes.GANConfig = datatypes.GANConfig()):
    in_preset = Input(shape=in_dim)
    noisy_input = GaussianNoise(
        parameters.discriminator_parameters.input_noise_stdev)(in_preset)
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    in_label = None
    if parameters.cgan_labels:
        # label input
        in_label = Input(shape=len(parameters.cgan_labels))
        li = Dense(len(parameters.cgan_labels),
                   kernel_initializer=init, kernel_constraint=const
                   )(in_label)
        li = LeakyReLU(alpha=parameters.discriminator_parameters.relu_alpha
                       )(li)
        li = Dropout(parameters.discriminator_parameters.dropout)(li)
        for _ in range(
                parameters.discriminator_parameters.category_input_layers):
            li = Dense(parameters.discriminator_parameters.category_layer_size,
                       kernel_initializer=init,
                       kernel_constraint=const)(li)
            li = LeakyReLU(alpha=parameters.discriminator_parameters.relu_alpha
                           )(li)
            li = Dropout(parameters.discriminator_parameters.dropout)(li)
        common_input = Concatenate()([noisy_input, li])
    else:
        common_input = noisy_input
    model = Dense(parameters.discriminator_parameters.layer_size,
                  kernel_initializer=init, kernel_constraint=const
                  )(common_input)
    model = LeakyReLU(alpha=parameters.discriminator_parameters.relu_alpha
                     )(model)
    model = Dropout(parameters.discriminator_parameters.dropout)(model)
    for _ in range(parameters.discriminator_parameters.hidden_layers):
        model = Dense(parameters.discriminator_parameters.layer_size,
                      kernel_initializer=init, kernel_constraint=const)(model)
        model = LeakyReLU(alpha=parameters.discriminator_parameters.relu_alpha
                          )(model)
        model = Dropout(parameters.discriminator_parameters.dropout)(model)
    model = Dense(1, activation='linear')(model)
    if parameters.cgan_labels:
        model = Model([in_preset, in_label], model)
        plot_model(model, 'crit.png', show_shapes=True)
    else:
        model = Model(in_preset, model)
    # compile model

    opt = RMSprop(lr=parameters.discriminator_parameters.learning_rate)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# define the standalone generator model
def define_generator(out_dim: int = 216,
                    parameters: datatypes.GANConfig = datatypes.GANConfig()):
    latent_input = Input(shape=parameters.generator_parameters.latent_dim)
    init = RandomNormal(stddev=0.02)
    in_label = None
    # weight initialization
    if parameters.cgan_labels:
        # label input
        in_label = Input(shape=len(parameters.cgan_labels))
        li = Dense(len(parameters.cgan_labels),
                   kernel_initializer=init,
                   )(in_label)
        li = LeakyReLU(alpha=parameters.generator_parameters.relu_alpha
                       )(li)
        li = Dropout(parameters.generator_parameters.dropout)(li)
        for _ in range(parameters.generator_parameters.category_input_layers):
            li = Dense(parameters.generator_parameters.category_layer_size,
                       kernel_initializer=init)(li)
            li = LeakyReLU(alpha=parameters.generator_parameters.relu_alpha
                           )(li)
            li = Dropout(parameters.generator_parameters.dropout)(li)
        common_input = Concatenate()([latent_input, li])
    else:
        common_input = latent_input
    model = Dense(parameters.generator_parameters.layer_size,
                  kernel_initializer=init)(common_input)
    model = LeakyReLU(alpha=parameters.generator_parameters.relu_alpha)(model)
    model = Dropout(parameters.generator_parameters.dropout)(model)
    for _ in range(parameters.generator_parameters.hidden_layers):
        model = Dense(parameters.generator_parameters.layer_size,
                      kernel_initializer=init)(model)
        model = LeakyReLU(
            alpha=parameters.generator_parameters.relu_alpha)(model)
        model = Dropout(parameters.generator_parameters.dropout)(model)
    model = Dense(out_dim, kernel_initializer=init,
                  activation='tanh')(model)
    if parameters.cgan_labels:
        model = Model([latent_input, in_label], model)
        plot_model(model, 'gen.png', show_shapes=True)
    else:
        model = Model(latent_input, model)
    return model


# define the combined generator and discriminator model, for updating the
# generator
def define_gan(g_model, d_model, config, learning_rate: float = 0.00005):
    # make weights in the discriminator not trainable
    d_model.trainable = False

    if config.cgan_labels:
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to
        # discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a
        # classification
        model = Model([gen_noise, gen_label], gan_output)
    else:
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)

    # compile model
    opt = RMSprop(lr=learning_rate)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# train the generator and discriminator
def train(g_model, c_model, gan_model, vectors, labels, dest_preset_path,
          config: datatypes.GANConfig = datatypes.GANConfig()):
    # calculate the number of batches per training epoch
    bat_per_epo = int(vectors.shape[0] / config.batch_size)
    # calculate the number of training iterations
    n_steps = bat_per_epo * config.num_epochs
    # calculate the size of half a batch of samples
    half_batch = int(config.batch_size / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = [], [], []
    # manually enumerate epochs
    for i in range(n_steps):
        real_critic_loss, fake_critic_loss, gen_loss = _run_epoch(c_model,
                                                                  config,
                                                                  g_model,
                                                                  gan_model,
                                                                  half_batch,
                                                                  labels,
                                                                  vectors)
        if verbose:
            print('>%d, c_real=%.3f, c_fake=%.3f gen=%.3f' % (
                i + 1, real_critic_loss, fake_critic_loss, gen_loss))
        c1_hist.append(real_critic_loss)
        c2_hist.append(fake_critic_loss)
        g_hist.append(gen_loss)
        # evaluate the model performance every 'epoch'
        if (i + 1) % 100 == 0:
            plot_history(c1_hist, c2_hist, g_hist, i, dest_preset_path)
            summarize_performance(i, g_model, dest_preset_path,
                                  config=config)
    plot_history(c1_hist, c2_hist, g_hist, 'final', dest_preset_path)


def _run_epoch(c_model, config,
               g_model, gan_model, half_batch, labels, vectors):
    c_real_losses, c_fake_losses = [], []
    for _ in range(config.discriminator_update_multiplier):
        real_loss = _update_critic_on_real_samples(c_model, config, half_batch,
                                                   labels, vectors)
        c_real_losses.append(real_loss)
        fake_loss = _update_critic_on_fake_samples(c_model, config, g_model,
                                                   half_batch)
        c_fake_losses.append(fake_loss)

    g_loss = _update_generator(config, gan_model, half_batch)
    return np.mean(c_real_losses), np.mean(c_fake_losses), g_loss


def _update_critic_on_real_samples(c_model, config, half_batch, labels, vectors):
    real_vectors, class_labels, real_labels = \
        _generate_real_samples(vectors, labels, n_samples=half_batch)
    real_loss = _update_critic(c_model, config, real_vectors, class_labels,
                               real_labels)
    return real_loss


# select real samples
def _generate_real_samples(vectors, labels, n_samples):
    # choose random instances
    ix = np.random.randint(0, vectors.shape[0], n_samples)
    # retrieve selected images
    selected_vectors = vectors[ix]
    # retrieve selected class labels
    selected_class_labels = labels[ix]
    # generate 'real' authenticity labels (1)
    selected_authenticity_labels = -np.ones((n_samples, 1))
    return selected_vectors, selected_class_labels, selected_authenticity_labels


def _update_critic(model, config, vectors, class_labels, auth_labels):
    if config.cgan_labels:
        critic_input = [vectors, class_labels]
    else:
        critic_input = vectors
    return model.train_on_batch(critic_input, auth_labels)


def _update_critic_on_fake_samples(c_model, config, g_model, half_batch):
    gen_vectors, class_labels, fake_labels = generate_fake_samples(
        g_model=g_model,
        latent_dim=config.generator_parameters.latent_dim,
        n_samples=half_batch,
        labels=config.cgan_labels
    )
    fake_loss = _update_critic(c_model, config, gen_vectors, class_labels,
                               fake_labels)
    return fake_loss


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples, labels=None):
    # generate points in latent space
    lat_input = _generate_latent_points(latent_dim, n_samples)
    # create fake authenticity labels for the generated samples
    fake_labels = np.ones((n_samples, 1))
    # create random cgan class labels
    if labels:
        class_labels = _generate_random_class_labels(n_samples, len(labels))
        # predict outputs
        gen_vectors = g_model.predict([lat_input, class_labels])
    else:
        # predict outputs
        gen_vectors = g_model.predict(lat_input)
        class_labels = fake_labels

    # normalize categories, booleans
    # if index is part of category or bool
    #  set max to one and others to zero
    key_paths = constants.TARGET_KEYS
    for vector in gen_vectors:
        category_adder = 0
        for key_index in range(len(key_paths)):
            i = key_index + category_adder
            key = key_paths[key_index][-1]
            if ableton_analog.is_boolean_param(key):
                vector[i] = normalize_boolean_parameter(vector[i])
            elif constants.CATEGORICAL_RANGES.get(key):
                cat_range = constants.CATEGORICAL_RANGES.get(key)
                category_adder += cat_range - 1
                value = np.argmax(vector[i:i + cat_range - 1])
                assert value in range(cat_range)
                for j in range(cat_range):
                    vector[i+j] = -1.
                vector[i+value] = 1.
                # print(f'{key_paths[key_index][-1]}: {vector[i]}')

    return gen_vectors, class_labels, fake_labels


def normalize_boolean_parameter(value):
    if round(value) < 0:
        return -1.
    else:
        return 1.


def _update_generator(config, gan_model, n_samples):
    gan_input = _generate_latent_points(
        config.generator_parameters.latent_dim, n_samples)
    # invert labels for the fake samples to update generator weights
    real_labels = -np.ones((n_samples, 1))
    # update the generator via the critic's error
    if config.cgan_labels:
        class_labels = _generate_random_class_labels(
            n_labels=n_samples, n_classes=len(config.cgan_labels))
        gan_input = [gan_input, class_labels]
    g_loss = gan_model.train_on_batch(gan_input, real_labels)
    return g_loss


# generate points in latent space as input for the generator
def _generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def _generate_random_class_labels(n_labels, n_classes):
    one_indicies = np.random.randint(low=0, high=n_classes, size=n_labels)
    labels = np.zeros((n_labels, n_classes))
    labels[np.arange(one_indicies.size), one_indicies] = 1
    return labels


# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, name, dest_dir):
    # plot history
    pyplot.plot(d1_hist, label='crit_real')
    pyplot.plot(d2_hist, label='crit_fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    # save plot to file
    plots_dir = os.path.join(dest_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    pyplot.savefig(os.path.join(plots_dir, f'{name}_plot_line_plot_loss.png'))
    pyplot.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, dest_preset_path,
                          n_samples=100, config=datatypes.GANConfig()):
    # prepare fake examples
    gen_vectors, class_labels, fake_labels = generate_fake_samples(
        g_model,
        config.generator_parameters.latent_dim,
        n_samples,
        config.cgan_labels
    )
    # rescale back to [0,1] to write the preset for human evaluation
    gen_presets = (gen_vectors + 1.0) / 2.0
    gen_presets = np.clip(gen_presets, 0, 1)
    presets_dir = os.path.join(dest_preset_path, 'presets')
    if not os.path.exists(presets_dir):
        os.makedirs(presets_dir)
    if config.cgan_labels:
        label_indices = np.argmax(class_labels, axis=1)
        class_labels = list(
            map(lambda label: config.cgan_labels[label], label_indices)
        )
    save_presets(gen_presets, epoch, presets_dir,
                 labels=class_labels if config.cgan_labels else None)
    # save the generator model tile file
    models_dir = os.path.join(dest_preset_path, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    filename = os.path.join(models_dir, f'generator_model_e{epoch + 1}.h5')
    g_model.save(filename)


# create and save a plot of generated images (reversed grayscale)
def save_presets(examples, epoch, path, n=1, labels=None):
    # write n presets
    for i in range(n):
        filename = f'generated_preset_{i}e{epoch+1}'
        if labels is not None:
            filename += labels[i]
        ableton_analog.write_preset_from_vector(examples[n],
                                                filename, path)
        if verbose:
            print(f'Wrote preset {filename}')
