#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import argparse
import os
import subprocess

import argcomplete


def list_choices(folder):
    def choices(prefix, parsed_args, **kwargs):
        cfgs_list = (c[:-5] for c in os.listdir(folder) if c.endswith('.json') and c.startswith(prefix))
        return cfgs_list
    return choices


parser = argparse.ArgumentParser(description='Template')
parser.add_argument('-m', '--model_config_path', type=str, required=True, metavar='/path/to/model/config.json', help='Path to model config .json file').completer = list_choices('res/models')
parser.add_argument('-d', '--dataset_config_path', type=str, required=True, metavar='/path/to/dataset/config.json', help='Path to dataset config .json file').completer = list_choices('res/datasets')
parser.add_argument('-r', '--run_config_path', type=str, required=True, metavar='/path/to/run/config.json', help='Path to run config .json file').completer = list_choices('res/run')
parser.add_argument('-g', '--gpus', type=str, default='0', metavar='cuda device, i.e. 0 or 0,1,2,3 or cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
parser.add_argument('-t', '--test_epoch', type=int, metavar='epoch want to test', help='epoch want to test')
parser.add_argument('--ci', action='store_true', default=False, help='running CI')
argcomplete.autocomplete(parser)
args = parser.parse_args()

process = [
    'python3', '-m', 'main',
    '-m', str(args.model_config_path),
    '-d', str(args.dataset_config_path),
    '-r', str(args.run_config_path),
    '-g', str(args.gpus),
]
if args.test_epoch is not None:
    process.append('-t')
    process.append(str(args.test_epoch))
if args.ci:
    process.append('--ci')
subprocess.run(process, check=True)
