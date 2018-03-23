Neural Networks for Overlap Muon Track Finder
==============

Bunch of useful (hope so) tools for neural network inference based on data from OMTF simulaion.
Research goal is to find optimal network architecture which could be concurrent to actualy implemented algorithm in CMS detector.

Content
--------------

Here is short description. I tried to keep my code well documented, so it's just a brief.

Package contains:

* `OMTFDataset` - handles data from Monte Carlo OMTF simulation, 
  set of TFRecord files, imports OMTF simulation data from `*.npz` files and saves as TFRecords files
* `OMTFNN` - wrapper for network and its metadata
* `OMTFNNStorage` - manager of OMTFNNs, helps keeping whole data in one place
* `OMTFRunner` - trains and tests prepared models, takes dataset and feeds model, returns some results
* `OMTFInputPipe` - creates customized pipe from provided dataset, helps with model feeding
* some utils - please, see `root_utils` and `utils` directory


Data available from OMTFInputPipe:

* hits array (full / reduced)
* production data:
  * pt label
  * sign label
  * pt original value
  * pt code
  * pt class

* OMTF data:
  * pt label
  * sign label
  * pt value

--------------

Any questions?  
Please, send me an email. :)

