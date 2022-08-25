1. System requirements:
	The software of DBlink was tested on Windows OS with python 3.8 installed. 
	We supply Anaconda environment as a YAML file, but the code is also runnable on PyCharm.
	No additional software is required for running DBlink.

2. Installation guide:
	To install DBlink, you should make sure you have at least python 3.8 and Anaconda 4.10.1 installed on your computer.
	Next, you should install DBlink's environment by opening the Anaconda prompt window and typing the following command:
	conda env create -f path/to/env/file.yml --name DBlink-env

	Typical installation time is 5-10 minutes.
	
3. Demo:
	To run a simple demo of DBlink on simulated data we have supplied three files:
	a. y_test - contains 4 simulated filament structures drifting and rotating over 3000 frames
	b. X_test - contains the simulated localization video of the same length for each one of the experiments
	c. LSTM_model - contains the model weights. The model was trained on simulated filament data

	The demo should run for ~1 minute if using a GPU and for ~8 minutes if using a CPU.
	
	Here is a short explanation on how to train and test DBlink on simulated/ experimental data.
	To run the following steps please go to demo directory and open demo.py file supplied in this directory.
	Step I: Parameter Initialization
		The run is controlled by four different flags:
			a. GenerateTrainData - whether you wish to generate new simulated training samples
			b. GenerateTestData - whether you wish to generate new simulated testing samples
			c. TrainNetFlag - whether you wish to train the network in the current run
			d. TestOnRealData - whether you wish to test the network on simulated data or experimental data (a csv file containing the localizations)
			
		For example: to test the network on simulated data, you should specify the following values:
			a. GenerateTrainData = False
			b. GenerateTestData = False (unless you wish to generate new test data)
			c. TrainNetFlag = False
			d. TestOnRealData = False
		
	Step II: Training data generation
		In this step you may generate training samples and validation samples for training purposes.
		In case you only wish to test the network, make sure you specified the TrainNetFlag as False to skip this step.
		
	Step III: Build Model, loss and optimizer
		In this step the model weights, the loss function and the optimizer are initialized.
		In case you only wish to test the network, change nothing in the parameter initialization.
		
	Step IV: Training the model 
		If you specified TrainNetFlag = True, network training would begin. This step necessitates a single available GPU, otherwise the run would take long time.
		We suggest first completing a single test run before trying to train the network.
		
	Step V: Testing the model
		If you specified TestOnRealData = True, you would have to supply a csv containing localizations of some SMLM video.
		Otherwise, the network would reconstruct the simulated data in the X_test, y_test files we have provided.
		
		The outputs of this step would be saved in a directory named ./tmp_results
		The outputs are two numpy files containing the ground truth video and the reconstructed video, and an additional mp4 video file visualizing both of them side by side.
		Expected outputs are located at the "expected outputs" directory provided for this demo.
		
4. Instructions for use
	To run DBlink on experimental data, you would have to first analyze your SMLM experiment using a localization software.
	Our csv data could be downloaded from ShareLoc website, at: XXXXXXXXXXXXXXXXXXXXXXXXXXXx
	
	The localization software output should be a csv file, containing at least three columns per detected emitter: detection frame, x position, and y position.
	Then, you should change the run flags in the demo.py file as shown here:
		a. GenerateTrainData = False
		b. GenerateTestData = False (unless you wish to generate new test data)
		c. TrainNetFlag = False
		d. TestOnRealData = True
	
	Next, you should update the path to the csv file and the csv file name.	In demo_exp_params.py you have a class named "demo_params". 
	In this class, you should change the variable self.path to contain the path to the csv file, and you should change the variable self.filename to contain the csv file name.
	
	Now you may run demo.py and obtain super spatiotemporal resolution reconstruction of your experimental data in mp4 format at the "tmp_results" directory.
	
Good luck!
	