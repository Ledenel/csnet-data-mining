from unittest import TestCase
import dataset as ds

class DatasetTest(TestCase):
    def test_dataset_load(self):
        loader = ds.CodeSearchDatasetLoader()
        samp = loader.get(language="ruby")[0]
        assert samp['language'] == "ruby"

        


