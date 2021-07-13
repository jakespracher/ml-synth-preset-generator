# Generating Musical Synthesizer Patches with Machine Learning

This repo accompanies this [blog post](https://jakespracher.medium.com/generating-musical-synthesizer-patches-with-machine-learning-c52f66dfe751). The premise is to generate high-quality presets for the Ableton Analog synthesizer in a particular style automatically using generative machine learning models.

## Setup

This repo currently requires Python 3.7.9 until Tensorflow can be migrated to version 2+.

Feel free to acquire this interpreter via your preferred method. One way to do so would be

1. [Install pyenv](https://github.com/pyenv/pyenv#installation)
2. `pyenv install 3.7.9`
3. `pyenv global 3.7.9`
4. `python -m venv venv`
5. `source venv/bin/activate`
6. `Pip install -r requirements.txt`

## Running Training

To run training you can use the `run_gan.py` script like this:

`python run_gan.py --config-file data/configs/config_240_cgan.json`

It also supports various CLI options including hyperparameter search. Run `python run_gan.py --help`

By default, this will output models, plots, and presets to data/generated in the project directory.

## Generating Presets Using the Model

To generate presets from a model, use the `generate_from_gan.py` script. Example usage:

`python generate_from_gan.py --config-file data/configs/config_240_cgan.json --model-path data/configs/config_240_cgan_generator_model_e1000.h5 --dest-path . --n-samples 10`

To evaluate presets, you will need [Ableton](ableton.com). You can get a 90 day free trial [here](https://www.ableton.com/en/trial/)

## Running Tests

Test coverage is currently very limited. To run the tests use `python -m pytest tests/`

## Architecture

This project uses the Keras framework (built in to Tensorflow) to construct and train the neural network.

The general flow is as follows:

1. Read training data directly from the repo at `data/analog_library` (theres not much data available sadly)
2. Parse presets into vectors (logic in `presetml/parsing/ableton_analog.py`)
3. Run training (logic in `generation/gan.py`)
