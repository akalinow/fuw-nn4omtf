# TO DO

## Notes
- OMTF & NN results can be compared only for given set of bins

- Initialization of variables
  - with batch normalization it is not the case anymore
## Done
* Stopping training with keyboard interrupt
* [OMTFNN]
  * Add additional requiremnt of special bin for `NOT KNOWN` state - `0`-pT bin edge
  * Special `NOT KNOWN` state for sign - required in case when no signal was registered=
* [INPUT PIPE] 
  * applying transformation to input data 
  * changed bucket-functions for `pt` and `sgn` data
* [NET BUILDER]
  * fully connected layer creator with batch normalization

## IN PROGRESS
* Upgrade statistics collection
  * Add extra softmax tensor at the end of net in `test` configuration
  * gethering lots data in `test`; `valid` and `train` should be bare
  * neuron activation analitycs
    * generation of 3D historgram, -> neuron outputs over dataset, | probability, / pt code
* cleanup `train` process 

## REQUIRED 

* add samples limits in `valid` and `test` process
* record important only
* record stats only in `valid` 
* save data summaries in csv
    - special tool should read and prepare correct plot
* megre many summaries before adding to TB 

* store only few of all statistics in network...

* Comparing models - take data from test runs and generate report

* Analytics tools (at least for input)
	* preview of weigths and activation patterns => understanding of learned features
* dry run option - runtime estimation?
* add accuracy/efficency presentation
  * add comparision with omtf algorithom
  * online version
	* show recent accuracy and compare it with omtf algorythom
  * offline - by dumping data into file?
    * create tools for dump analiytics
    * histograms ---^ analitycs

* utilities for:
    * showing up probabilities at the end
 
* data quantization -> FPGA
  * in case of having good network and implementing it on fpga, it probably would be good idea to
    quantize states but... it's hard??

* passing parameters in omtfrunner
  * changing optimizer
  * changing pipeline 
    * interleave
       * cycle
       * reps
* remove old net in existing directory (or ask)
* outputing generated data to file (exact numbers, csv like?)
  * outputing levels?
    * rather binary format than ascii
    * eventualy converter in tools to generate cvs


## NICE TO HAVE

* tools for remembering venv paths and activating?
  * env autosetup??
* possibility of dumping model, getting pure varables in bytes or whatever
