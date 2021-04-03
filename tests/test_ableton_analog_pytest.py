import pytest
from presetml.parsing import ableton_analog
from presetml import constants


@pytest.fixture(
    params=ableton_analog.read_presets_from_dir(constants.TEST_PRESETS_PATH)[:10]
)
def test_preset_data(request):
    return request.param


def test_read_rewrite_should_generate_identical_presets(test_preset_data):
    preset, vector, _ = test_preset_data
    xml = ableton_analog.generate_xml_from_vector(vector)
    assert xml.encode() == preset
