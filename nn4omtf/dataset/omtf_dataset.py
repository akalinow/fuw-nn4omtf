from os import path


class OMTFDataset:
    """Dataset paths and metadata.
    It's used for mapping and file paths extracting.
    """
    
    HITS_TYPE_FULL = "FULL"
    HITS_TYPE_REDUCED = "REDUCED"

    def __init__(self, path):
        self.root_path = path
        if not path.exists(path):
            raise FileNotFoundError("")


    def _load(self):
        pass


    def __str__(self):
        pass

    
    def _read_descriptor(self):
        pass


    def _save_descriptor(self):
        pass

    
    def get_dataset(self, name='train', ptc_min=None, ptc_max=None):
        """
        Returns list of paths which coresponds to events with muons' pT code
        between given limits.
        """
        pass

    
    def get_hits_type(self):
        """Returns which tensor of hits was saved in dataset."""
        return self.hits_type
