# TO DO

## Notes
- OMTF & NN results can be compared only for given set of bins

## Required

* [IN PROGRESS] Prepare output for +/- charge sign classification

* Test and compare procedure
	* Testing - done
	* Comparing models - take data from test runs and generate report
* [IN PROGRESS] Upgrade statistics collection

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


* neuron activation analitycs

* tools for remembering venv paths and activating?
  * env autosetup??
 
* data quantization -> FPGA
  * in case of having good network and implementing it on fpga, it probably would be good idea to
    quantize states but... it's hard??

* passing parameters in omtfrunner
  * changing optimizer
  * changing pipeline 
    * interleave
       * cycle
       * reps
    * batchsize
    * reps
* repair train logging
* remove old net in existing directory (or ask)
* outputing generated data to file (exact numbers, csv like?)
  * outputing levels?
    * rather binary format than ascii
    * eventualy converter in tools to generate cvs
* Initialization of variables
  * Read paper about best practises 
  * dispersion should depend on input dimensions


## NICE TO HAVE

* possibility of dumping model, getting pure varables in bytes or whatever
