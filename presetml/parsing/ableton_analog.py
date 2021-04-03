import os
import xmltodict
import gzip
import json
import uuid
import numpy as np
from typing import NoReturn

from presetml import constants

EXTRACTED_PRESET_DIR = '/tmp/presets'


def read_presets_from_dir(
        target_path=f'{constants.ANALOG_PRESETS_PATH}',
        labels=None) -> [str, np.array]:
    vectors = []
    label = None
    for root, dirs, files in os.walk(target_path):
        if labels:
            found = False
            for key in labels:
                if key in root:
                    # build one hot vector
                    found = True
                    label = int(labels.index(key))
            if not found:
                continue
        vectors += [vector_from_file(root, f, constants.TARGET_KEYS) + (label, )
                    for f in files if f.endswith('.adv')]
    return vectors


def vector_from_file(root: str, file_name: str,
                     target_keys: list = constants.TARGET_KEYS
                     ) -> (str, np.array):
    xml_preset = unzip_preset(os.path.join(root, file_name))
    return xml_preset, vector_from_preset(xml_preset, target_keys)


def unzip_preset(file_path: str) -> str:
    with gzip.open(file_path, 'r') as zip_handle:
        return zip_handle.read()


def vector_from_preset(xmlpreset: str, target_keys: list) -> np.array:
    parameters_json = xmltodict.parse(xmlpreset)
    return build_vector(parameters_json, target_keys)


def build_vector(parameters_json: dict, target_keys: list) -> np.array:
    parameters = [(key[-1], extract_target_metadata(parameters_json, key))
                  for key in target_keys]
    parameter_values = []
    for key, metadata_dict in parameters:
        parameter_values += extract_values_from_parameter(key, metadata_dict)
    return list(parameter_values)


def extract_target_metadata(parameters_json, target_key):
    value_dict = parameters_json
    for level in target_key:
        value_dict = value_dict[level]

    return value_dict


def extract_values_from_parameter(key, metadata_dict):
    if not metadata_dict:
        raise Exception('None parameter as input')
    param_value_string = get_param_value_string(metadata_dict)
    if key in constants.CATEGORICAL_RANGES:
        param_values = expand_categorical_variable(key, metadata_dict,
                                                   param_value_string)
    elif param_value_string == 'true':
        param_values = [1]
    elif param_value_string == 'false':
        param_values = [0]
    else:
        param_values = [extract_value_from_string(param_value_string,
                                                      metadata_dict)]
    return param_values


def get_param_value_string(param) -> str:
    param_value = param.get('@Value')
    try:
        return param_value if param_value else param['Manual']['@Value']
    except KeyError:
        raise Exception('Un-parseable parameter provided')


def expand_categorical_variable(key, metadata_dict, param_value_dict):
    value = extract_value_from_string(param_value_dict, metadata_dict)
    param_value = [0] * constants.CATEGORICAL_RANGES.get(key, 2)
    param_value[value] = 1
    return param_value


def extract_value_from_string(param_value_dict, param_metadata_dict):
    param_value_dict = extract_number_from_string(param_value_dict)
    value_range = extract_range(param_metadata_dict)
    if value_range:
        param_value_dict = normalize_parameter(param_value_dict, value_range)
    return param_value_dict


def extract_number_from_string(string):
    if string == 'true':
        param_value = 1
    elif string == 'false':
        param_value = 0
    else:
        try:
            param_value = int(string)
        except ValueError:
            param_value = float(string)
    return param_value


def extract_range(param):
    value_range = param.get('MidiControllerRange')
    if value_range:
        return extract_number_from_string(value_range['Min']['@Value']), \
               extract_number_from_string(value_range['Max']['@Value'])
    else:
        return None


def normalize_parameter(value, range):
    return (value - range[0]) / (range[1] - range[0])


def extract_value_from_string(param_value_dict, param_metadata_dict):
    param_value_dict = extract_number_from_string(param_value_dict)
    value_range = extract_range(param_metadata_dict)
    if value_range:
        param_value_dict = normalize_parameter(param_value_dict, value_range)
    return param_value_dict


def write_preset_from_vector(vector: np.array, name: str = None,
                             path: str = None, key_paths: list = None,
                             default_dict: dict = None) -> NoReturn:
    if name is None:
        name = f'test-preset-{uuid.uuid4()}'
    if path is None:
        path = constants.GENERATED_PRESETS_DIR
    xml = generate_xml_from_vector(vector, key_paths, default_dict)
    write_zipped_preset(os.path.join(path, f'{name}.adv'), xml)


def generate_xml_from_vector(vector: list, key_paths: list = None,
                             default_dict: dict = None) -> str:
    if default_dict is None:
        default_dict = get_default_preset_dict()
    if key_paths is None:
        key_paths = constants.TARGET_KEYS
    category_adder = 0
    preset_dict = default_dict
    for key_index in range(len(key_paths)):
        i = key_index + category_adder
        key = key_paths[key_index][-1]
        if is_boolean_param(key):
            value = round(vector[i])
            assert value in [0, 1]
        elif constants.CATEGORICAL_RANGES.get(key):
            cat_range = constants.CATEGORICAL_RANGES.get(key)
            category_adder += cat_range - 1
            value = np.argmax(vector[i:i + cat_range - 1])
            assert value in range(cat_range)
        else:
            value = vector[i]
            assert 0 <= value <= 1

        target_dict = extract_target_metadata(preset_dict,
                                              key_paths[key_index][:-1])
        target_dict[key] = write_metadata_for_parameter(target_dict[key], value)
    xml = xmltodict.unparse(preset_dict)
    return xml


def get_default_preset_dict() -> dict:
    with open(constants.DEFAULT_PRESET_JSON_PATH, 'r') as file:
        return json.loads(file.read())


def is_boolean_param(key):
    bools = ['ExponentialSlope', 'LFOGateReset', 'FreeRun', 'FilterSlave', 'KeyboardUnison', 'OscillatorMode']
    if 'Toggle' in key \
            or 'Legato' in key \
            or key in bools:
        return True
    else:
        return False


def write_metadata_for_parameter(metadata_for_key, value) -> dict:
    value_range = extract_range(metadata_for_key)
    if value_range:
        value = denormalize_parameter(value, value_range)
        assert value_range[0] <= value <= value_range[1]

    translated_value = write_value_for_parameter(metadata_for_key, value)
    if metadata_for_key.get('@Value'):
        metadata_for_key['@Value'] = translated_value
    else:
        metadata_for_key['Manual']['@Value'] = translated_value

    return metadata_for_key


def denormalize_parameter(value, range):
    return value * (range[1] - range[0]) + range[0]


def write_value_for_parameter(key, value) -> str:
    if is_boolean_param(key):
        return 'true' if value else 'false'
    else:
        towrite = str(value)
        if towrite.endswith('.0'):
            towrite = towrite[:-2]
        return towrite


def write_zipped_preset(file_path: str, content: str) -> NoReturn:
    with gzip.open(file_path, 'wb') as zip_handle:
        zip_handle.write(content.encode('utf-8'))


def write_preset_from_xml(xml_path, target_path):
    with open(xml_path, 'r') as xml:
        write_zipped_preset(target_path, xml.read())