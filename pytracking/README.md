# PyTracking

A general python library for visual tracking algorithms. 

## Running a tracker
The installation script will automatically generate a local configuration file  "evaluation/local.py". In case the file was not generated, run ```evaluation.environment.create_default_local_file()``` to generate it. Next, set the paths to the datasets you want
to use for evaluations. You can also change the path to the networks folder, and the path to the results folder, if you do not want to use the default paths. If all the dependencies have been correctly installed, you are set to run the trackers.  

**Run the tracker on some dataset sequence**  
This is done using the run_tracker script. 
```bash
python run_tracker.py tracker_name parameter_name --dataset dataset --sequence sequence --debug debug --threads threads
```  
Here, the dataset_name is the name of the dataset used for evaluation, e.g. ```otb```. See [evaluation.datasets.py](evaluation/datasets.py) for the list of datasets which are supported. The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'Soccer'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

Take CIA18 as an example.
```bash
python run_tracker.py CIA CIA18 --dataset lasot
```

## Visdom

All trackers support [Visdom](https://github.com/facebookresearch/visdom) for debug visualizations. To use visdom, start the visdom
server from a seperate command line: 

```bash
visdom
```  

Run the tracker with the ```debug``` argument > 0. The debug output from the tracker can be 
accessed by going to ```http://localhost:8097``` in your browser. Further, you can pause the execution of the tracker,
or step through frames using keyboard inputs. 

![visdom](.figs/visdom.png)
 
