from nn4omtf.model import OMTFModel
from nn4omtf.pipe import OMTFInputPipe
from nn4omtf.dataset import OMTFDataset
from nn4omtf.runner import OMTFRunner

def import_root_utils():
    """Import part of package which uses ROOT."""
    from . import root_utils
