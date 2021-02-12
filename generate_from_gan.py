import click
import numpy as np
import json
import cattr
from presetml.generation import gan
from presetml.parsing import ableton_analog
from presetml import constants, datatypes


@click.command()
@click.option('--n-samples', default=1)
@click.option('--model-path')
@click.option('--config-file')
@click.option('--data-len', default=len(constants.TARGET_KEYS))
@click.option('--dest-path',
              default=constants.GENERATED_PRESETS_DIR)
@click.option('--generated-preset-name', default='generated_preset')
def main(n_samples, model_path, data_len, dest_path, generated_preset_name,
         config_file):
    with open(config_file, 'r') as f:
        config = json.loads(f.read())
        config = cattr.structure(config, datatypes.GANConfig)
    if config is None:
        raise Exception(f'Could not load {config_file}')

    g_model = gan.define_generator(data_len, config)
    g_model.load_weights(model_path)
    gen_vectors, class_labels, fake_labels = gan.generate_fake_samples(
        g_model,
        config.generator_parameters.latent_dim,
        n_samples,
        config.cgan_labels
    )
    if config.cgan_labels:
        class_labels = list(
            map(
                lambda label: config.cgan_labels[np.argmax(label)], # convert one-hot to int with argmax
                class_labels
            )
        )
    # rescale back to [0,1] to write the preset for human evaluation
    gen_presets = (gen_vectors + 1.0) / 2.0
    gen_presets = np.clip(gen_presets, 0, 1)
    for i in range(n_samples):
        ableton_analog.write_preset_from_vector(
            gen_presets[i],
            f'{generated_preset_name}_{i}_{class_labels[i]}',
            dest_path)


if __name__ == '__main__':
    main()
