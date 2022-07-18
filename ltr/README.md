# LTR

The way to train CIA tracker.

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate pytracking
python run_training.py train_module train_name
```
Here, ```train_module``` is the sub-module inside ```train_settings``` and ```train_name``` is the name of the train setting file to be used.

For example, you can train using the CIA18 settings by running:
```bash
python run_training CIA CIA18
```
