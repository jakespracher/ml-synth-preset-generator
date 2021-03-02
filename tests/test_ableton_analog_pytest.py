from presetml.parsing import ableton_analog
from presetml import constants


def test_read_rewrite_should_generate_identical_presets():
    for preset, vector, _ in ableton_analog.read_presets_from_dir(
            constants.TEST_PRESETS_PATH)[:10]:
        xml = ableton_analog.generate_xml_from_vector(vector)
        assert xml.encode() == preset
