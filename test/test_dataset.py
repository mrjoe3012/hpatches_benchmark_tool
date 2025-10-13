import os
from hpatches_benchmark.dataset.hpatches import HPatches
import logging

logging.basicConfig(level='INFO')

def test_dataset():
    try:
        dataset_path = os.environ['HPATCHES_DATASET_PATH']
    except KeyError:
        raise RuntimeError("HPATCHES_DATASET_PATH environment variable was unset!")
    hpatches = HPatches.load_hpatches(dataset_path) 
    expected_n_seq = 116
    assert len(hpatches.image_sets) == expected_n_seq, "Dataset was not loaded properly."
