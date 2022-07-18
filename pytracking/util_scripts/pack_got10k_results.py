import numpy as np
import os
import shutil
from pytracking.evaluation.environment import env_settings
import argparse

import pdb


def pack_got10k_results(tracker_name, param_name, output_name):
    """ Packs got10k results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().got_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        output_name - name of the packed zip file
    """
    output_path = os.path.join(env_settings().got_packed_results_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_path = env_settings().results_path
    for i in range(1,181):
        seq_name = 'GOT-10k_Test_{:06d}'.format(i)

        seq_output_path = '{}/{}'.format(output_path, seq_name)
        if not os.path.exists(seq_output_path):
            os.makedirs(seq_output_path)

        for run_id in range(3):
            '''
            res = np.loadtxt('{}/{}/{}/{}.txt'.format(results_path, tracker_name, param_name, seq_name))
            times = np.loadtxt(
                '{}/{}/{}/{}_time.txt'.format(results_path, tracker_name, param_name, seq_name),
                dtype=np.float64)
            '''
            res_file = '{}/{}/{}/{}.txt'.format(results_path, tracker_name, param_name, seq_name)
            times_file = '{}/{}/{}/{}_time.txt'.format(results_path, tracker_name, param_name, seq_name)
            res = []
            times = []
            with open(res_file, 'r') as f:
                data = f.readlines()
                for line in data:
                    line = eval(line.strip().replace('\t', ','))
                    res.append(line)
            with open(times_file, 'r') as f:
                time = f.readlines()
                for line in time:
                    line = eval(line.strip())
                    times.append(line)

            res = np.array(res)
            times = np.array(times)

            np.savetxt('{}/{}_{:03d}.txt'.format(seq_output_path, seq_name, run_id+1), res, delimiter=',', fmt='%f')
            np.savetxt('{}/{}_time.txt'.format(seq_output_path, seq_name, run_id+1), times, fmt='%f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)

def main():
    parser = argparse.ArgumentParser(description='Pack the tracking results on GoT-10K dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracker')
    parser.add_argument('param_name', type=str, help='Name of parameter file')
    parser.add_argument('output_name', type=str, help='Name of output zip file')

    args = parser.parse_args()

    pack_got10k_results(args.tracker_name, args.param_name, args.output_name)


if __name__ == '__main__':
    main()
