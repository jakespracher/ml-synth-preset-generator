import click
import xmltodict
import gzip
import re


def unzip_preset(file_path):
    with gzip.open(file_path, 'r') as zip_handle:
        return zip_handle.read()

TEST_FILE = '/Users/vader/Music/Ableton/User Library/Presets/Instruments/Analog/Leads/Wide Lead.adv'
TEST_GENERATED_FILE = '/Users/vader/analog-preset-ml/lev1keys'


@click.command()
@click.option('--preset-file',
              prompt='The target preset file',
              default=TEST_FILE)
@click.option('--output-file',
              prompt='Destination for generated source',
              default=TEST_GENERATED_FILE)
@click.option('--class-name',
              prompt='Class name for generating preset',
              default='AbletonAnalogPreset')
def run(preset_file, output_file, class_name):
    unzipped = unzip_preset(preset_file)
    preset_json = xmltodict.parse(unzipped)
    with open(output_file, 'w') as generated_source:
        generate_preset_class(preset_json['Ableton']['UltraAnalog']['SignalChain1']['Envelope.0'], generated_source, class_name)


def generate_preset_class(preset_dict, generated_source, class_name):
    # for key in attr dict, lookup key in preset dict, swap value.
    # functionality we want: given a preset, extract a vector for a set of keys
    # -> function takes a list of the keys as input, extracts the values, writes a list of vector
    # also, given a vector and keys, generate a preset that overrides the defaults.
    # -> function takes a vector and list of keys, loads default dict and overrides keys with vector values.
    for key in preset_dict.keys():
        generated_source.write(f'   ["Ableton", "UltraAnalog", "SignalChain1", "Envelope.0", "{key}"],\n')


if __name__ == '__main__':
    run()
