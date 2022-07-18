import numpy as np 
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList 
from pytracking.utils.load_text import load_text 
import os

import pdb


class TNL2KDataset(BaseDataset):
    """TNL2K dataset
    """
    def __init__(self, split):
        super().__init__()
        # Split can be test or train
        split_path = 'TNL2K_{}_subset'.format(split)
        self.base_path = os.path.join(self.env_settings.tnl2k_path, split_path)

        self.sequence_list = self._get_sequence_list()
        self.split = split 

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith('.jpg') or frame.endswith('.png')]
        try:
            try:
                frame_list.sort(key=lambda f: int(f.split('.')[0]))
            except:
                frame_list.sort(key=lambda f: int(f.split('_')[-1].split('.')[0] + \
                                                  f.split('_')[-1].split('.')[1]))
        except:
            frame_list.sort(key=lambda f: f.split('.')[0])
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list 
