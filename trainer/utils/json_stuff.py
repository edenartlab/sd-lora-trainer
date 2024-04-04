import ujson
import os


def save_as_json(dictionary_or_list, filename: str):
    with open(filename, "w") as fp:
        ujson.dump(dictionary_or_list, fp, indent=4)


def load_json(filename: str):
    assert os.path.exists(filename), f"Could not find json file: {filename}"
    with open(filename) as json_file:
        data = ujson.load(json_file)
    return data