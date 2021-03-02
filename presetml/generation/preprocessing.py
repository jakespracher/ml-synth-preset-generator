import numpy as np
from presetml import constants
from presetml.parsing import ableton_analog


def load_data(path=constants.ANALOG_PRESETS_PATH,
              label_names=None) -> (np.array, np.array):
    vectors, labels = [], []
    for _, vector, label \
            in ableton_analog.read_presets_from_dir(path, label_names):
        if label_names:
            one_hot_label = [0] * len(label_names)
            one_hot_label[label] = 1
        else:
            one_hot_label = label
        vector = np.array(vector)
        vectors.append(vector)
        labels.append(one_hot_label)

    return np.array(vectors).astype('float32'), np.array(labels)
