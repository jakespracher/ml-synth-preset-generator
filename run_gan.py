import click
import os
import attr
import cattr
import json
import itertools
import numpy as np
from joblib import Parallel, delayed

import presetml.generation.util
from presetml.generation import gan
from presetml import constants, datatypes


@click.command()
@click.option("--num-epochs", default=100)
@click.option("--batch-size", default=25)
@click.option("--search", is_flag=True)
@click.option("--cgan", is_flag=True)
@click.option("--config-file", default=None)
@click.option("--training-presets-path", default=constants.ANALOG_PRESETS_PATH)
@click.option("--generated-presets-dest", default=constants.GENERATED_PRESETS_DIR)
def main(
    num_epochs,
    batch_size,
    search,
    cgan,
    config_file,
    training_presets_path,
    generated_presets_dest,
):
    labels = constants.CGAN_LABELS if cgan else None
    if search:
        # runs preconfigured hyperparameter search
        if cgan:
            raise Exception("CGAN must be configured in search code")
        dataset = load_data(training_presets_path, batch_size)
        run_search(dataset, generated_presets_dest)
    elif config_file:
        with open(config_file, "r") as f:
            config = json.loads(f.read())
            labels = config.get("cgan_labels")
            dataset = load_data(training_presets_path, batch_size, labels)
            config["num_epochs"] = num_epochs
            config["batch_size"] = batch_size
            config = cattr.structure(config, datatypes.GANConfig)
            gan.run(dataset, generated_presets_dest, config)
    else:
        config = datatypes.GANConfig(
            batch_size=batch_size, num_epochs=num_epochs, cgan_labels=labels
        )
        dataset = load_data(training_presets_path, batch_size, labels)
        gan.run(dataset, generated_presets_dest, config)


def load_data(training_presets_path, batch_size, labels=None):
    vectors, labels = presetml.generation.util.load_real_samples(
        training_presets_path, labels
    )
    print(f"Loaded {len(vectors)} samples. Shape: {vectors.shape}")
    if batch_size > vectors.shape[0]:
        print("Batch size must be less than len of dataset")
    return vectors, labels


def run_search(dataset: np.array, dest_preset_path: str):
    search = {
        "num_epochs": [500],
        "batch_size": [25],
        "discriminator_update_multiplier": range(2, 5),
        "discriminator_parameters": {
            "learning_rate": [0.0001, 0.00005, 0.00001],
            "hidden_layers": range(2, 10, 2),
            "layer_size": range(100, 1000, 200),
            "dropout": [0.4],
            "relu_alpha": [0.2],
        },
        "generator_parameters": {
            "learning_rate": [0.0001, 0.00005, 0.00001],
            "latent_dim": range(100, 300, 100),
            "hidden_layers": range(2, 10, 2),
            "layer_size": range(100, 1000, 200),
            "dropout": [0.4],
            "relu_alpha": [0.2],
        },
    }

    search = [
        cattr.structure(config, datatypes.GANConfig)
        for config in generate_config_combinations(search)
    ]
    Parallel(n_jobs=4)(
        delayed(run_indiv_search)(search[i], i, len(search), dest_preset_path, dataset)
        for i in range(len(search))
    )


def run_indiv_search(config, i, total, dest_preset_path, dataset):
    global verbose
    verbose = False
    print(f" ------ Config {i}/{total} ------ ")
    iteration_dir = os.path.join(dest_preset_path, f"config_{i}")
    os.makedirs(iteration_dir)
    config_info = os.path.join(iteration_dir, "hyperparameter_config.json")
    with open(config_info, "w") as f:
        f.write(json.dumps(attr.asdict(config)))
    gan.run(dataset, iteration_dir, config)


def generate_config_combinations(search_dict):
    return [
        dict(zip(search_dict.keys(), value))
        for value in itertools.product(*_flatten_search_dict(search_dict))
    ]


def _flatten_search_dict(search_dict):
    return [
        generate_config_combinations(val) if isinstance(val, dict) else val
        for val in search_dict.values()
    ]


if __name__ == "__main__":
    main()
