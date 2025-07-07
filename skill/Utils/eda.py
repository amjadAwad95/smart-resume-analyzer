import json
from pathlib import Path
from collections import Counter
from itertools import chain


# load the data
def load_data(data: Path):
    with open(data, 'rb') as f:
        data = json.load(f)

    return data


def count_frequency(data: list):
    values_per_label = []
    for i in data:
        annotation = i['annotation']
        for annot in annotation:
            for key, value in annot.items():
                if key == 'label':
                    # collect values and append to a list
                    values_per_label.append({'label': value})
    # count all the values
    count = Counter(chain.from_iterable(elem.values() for elem in values_per_label))

    return count


data_path = r"../Data/ner.json"
data = load_data(data_path)
labels_count = count_frequency(data)
print(labels_count)


