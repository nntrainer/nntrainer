"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.

@file build_windows_resources.py
@date 16 July 2025
@brief Extract symbols from obj files to build def file
@author Grzegorz Kisala <gkisala@gmail.com>
"""
import sys
import os
import re
import subprocess
import argparse
from collections import defaultdict

def extract_syms(symfile, defs):
    print('Opening ', symfile)
    with open(symfile, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            line = re.sub(r'notype \(\)', 'func', line)
            line = re.sub(r'notype', 'data', line)
            pieces = line.split()
            
            if not pieces[0] or not re.match('^[A-F0-9]{3,}$', pieces[0]):
                continue

            try:
                symbol_name = pieces[6]
            except IndexError:
                continue
            
            if pieces[2] == "UNDEF":
                continue
            
            if pieces[4] != "External":
                continue
            
            if re.match('^@', symbol_name) or re.match('^[(]', symbol_name):
                continue
            
            if re.match('^__real', symbol_name) or re.match('^__xmm', symbol_name):
                continue
            
            if re.match('^__imp', symbol_name):
                continue
            
            if re.search('NULL_THUNK_DATA$', symbol_name) or re.match('^__IMPORT_DESCRIPTOR', symbol_name) or re.match('^__NULL_IMPORT', symbol_name):
                continue
            
            if re.match('^\\?\\?_C', symbol_name):
                continue
            
            defs[symbol_name] = pieces[3]

def writedef(deffile, defs):
    with open(deffile, 'w') as fh:
        fh.write("EXPORTS\n")
        for k in sorted(defs.keys()):
            isdata = defs[k] == 'data'

            if isdata:
                fh.write("  {} DATA\n".format(k))
            else:
                fh.write("  {}\n".format(k))

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects_dir', type=str, required=True, help='Path to directory with obj files')
    parser.add_argument('--working_dir', type=str, required=True, help='Path to working directory')
    return parser.parse_args()


def split_files(files):
    length = len(' '.join(files))
    if length > 30000:
        half = len(files) // 2
        return [*split_files(files[:half]), *split_files(files[half:])]
    else:
        return [files]

def run_command(args, files, cwd, exit_on_failure=True):
    # On Windows, for some reason, when long paths are disabled:
    # https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#registry-setting-to-enable-long-paths
    # There is a limit to how many files/how long the command line passed to dumpbin can be
    # This limit seems to be 32 768, but I'm not sure whether it's for sum of files or for all of command line arguments
    # To remedy this problem we split files so that each batch of concatenated files is no longer than 30k characters
    # For each batch we run dumpbin and then concatenate the results
    out = bytearray()
    split = split_files(files)
    for s in split:
        command = args + s
        print('Running command: {}'.format(command))
        ret = subprocess.run(command, cwd=cwd, check=exit_on_failure, capture_output=True)
        out.extend(ret.stdout)
        returncode = ret.returncode

        if exit_on_failure and returncode != 0:
            print('Application {} failed with returncode {}. Exiting...'.format(
                args[0], returncode))
            sys.exit(returncode)

    return out

def list_files(path, extension):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and file_path.endswith(extension):
            file_list.append(file_path)
    
    return file_list

def main():
    args = get_args()

    print("args.objects_dir: {}".format(args.objects_dir))
    sym_file = os.path.join(args.working_dir, 'nntrainer.sym')
    def_file = os.path.join(args.working_dir, 'nntrainer.def')
    files = list_files(args.objects_dir, '.obj')

    dumpbin_command = ['dumpbin', '/symbols']

    out = run_command(dumpbin_command, files, REPO_DIR)
    with open(sym_file, 'wb') as f:
        f.write(out)

    defs = defaultdict()
    extract_syms(sym_file, defs)

    writedef(def_file, defs)

if __name__ == '__main__':
    main()
