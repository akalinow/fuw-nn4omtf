Neural Networks for Overlap Muon Track Finder
==============

Jacek Łysiak - jacek.lysiako.o-at-gmail.com

Descripton
--------------

Bunch of useful (hope so) tools for neural network inference based on data from OMTF simulaion.
Research goal is to find optimal network architecture which could be concurrent to actualy implemented algorithm in CMS detector.

Content
--------------

Package contains:

* OMTFDataset - handles data from Monte Carlo OMTF simulation, set of TFRecord files
* OMTFNN - wrapper for network and its metadata
* OMTFNNStorage - manager of OMTFNNs, helps keeping whole data in one place
* OMTFNNTrainer - trains prepared models
* OMTFDatasetGenerator - imports data from OMTF simulations and saves as TFRecords files

To do...
--------------

* Saving model, trained data in OMTFNN
* Reseting/clearing trained model (cloning?)
* Possibility of dumping model, getting pure varables in bytes or whatever
* saving runs statistics for tb localy, variables and model should be small enougth to push on repo, dataset could be available on cloud
