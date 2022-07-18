import os 

class EnvironmentSettings:
    def __init__(self):
        path = os.getcwd()
        self.workspace_dir = path     # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = 'Path to your lasot dataset'
        self.got10k_dir = 'Path to your got10k training set'
        self.trackingnet_dir = 'Path to your trackingnet dataset'
        self.coco_dir = 'Path to your coco dataset'
