from unittest import TestCase
import dataset as ds

class DatasetTest(TestCase):
    def test_dataset_load(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        loader = ds.CodeSearchDatasetLoader()
        samp = loader.get(language="ruby")[0]
        assert samp.language == "ruby"

    def test_cache(self):
        loader = ds.CodeSearchDatasetLoader(max_chunks_in_memory=2)
        get_func = loader.pool.get_func.storage.backend
        dataset = loader.get(language="python")
        assert get_func.cache_info().hits == 0
        samp1 = dataset[0]
        samp2 = dataset[3]
        hits = get_func.cache_info().hits
        assert hits > 0 
        samp3 = dataset[120009]
        assert get_func.cache_info().hits == hits
        samp4 = dataset[40005]  # released LRU chunk 0
        assert get_func.cache_info().hits == hits
        samp5 = dataset[6]
        assert get_func.cache_info().hits == hits

        


