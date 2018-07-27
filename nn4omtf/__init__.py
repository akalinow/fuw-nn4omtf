from nn4omtf.statistics import OMTFStatistics
from nn4omtf.pipe import OMTFInputPipe
from nn4omtf.model import OMTFModel
from nn4omtf.dataset import OMTFDataset
from nn4omtf.runner import OMTFRunner
from nn4omtf.plotter import OMTFPlotter
from nn4omtf.const_files import FILE_TYPES

def import_root_utils():
    """Import part of package which uses ROOT."""
    from . import root_utils
