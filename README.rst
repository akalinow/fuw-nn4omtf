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
* `OMTFInputPipe` - creates customized pipe from provided dataset, helps with model feeding, provides additional data which helps with NN & OMTF comparison
* `OMTFStatistics` - data container
* some utils - please, see `root_utils` and `utils` directory


--------------

Any questions?  
Please, send me an email. :)

