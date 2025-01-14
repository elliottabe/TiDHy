# TiDHy: Timescale Demixing via Hypernetworks


## Setup for installing conda environment and dependencies
To install the repo there is a conda environment that will install the necessary packages. Make sure you are in the TiDHy Github directory.  
Use command:  
`conda env create -f environment.yaml`

After installing activate the conda environment:  
`conda activate TiDHy`

After pytorch is correctly installed run this command to install pip reqruirements:  
`pip install -r requirements.txt`

If the requirements.txt file does not install pytorch with cuda, go to this site to install the appropriate pytorch version:  
https://pytorch.org/get-started/locally/

To install TiDHy, in the repo folder use:  
`pip install -e .`

For SLDS comparison install ssm package:  
https://github.com/lindermanlab/ssm


## Example Code
To Train TiDHy you can run the Run_TiDHy.py script from terminal with hydra overrides:  
`python Run_TiDHy.py Run_TiDHy.py dataset=SLDS dataset.train.gpu=0 version=Example`


## Custom Datasets
To add a custom dataset you can load data in anyway you want. The final formatting should follow the convension of:  
- train_data: `(time x features)`  
- val_data:   `(time x features)`  
- test_data:  `(time x features)`  

The data can then be stacked with overlapping windows using the `stack_data` fucntion:  
`train_inputs = stack_data(train_inputs,cfg.train.sequence_length,overlap=cfg.train.sequence_length//cfg.train.overlap_factor)`
