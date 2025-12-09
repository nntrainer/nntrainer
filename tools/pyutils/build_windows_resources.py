"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.

@file build_windows_resources.py
@date 13 June 2025
@brief Build and package third party required for windows msvc build
@author Grzegorz Kisala <gkisala@gmail.com>
"""

import argparse
import os
import shutil
import sys
import subprocess

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
SUBPROJECTS_DIR = os.path.join(REPO_DIR, 'subprojects')
RESOURCES_DIR = os.path.join(REPO_DIR, 'nntrainer-windows-resource')

CLBLAST_DIR = os.path.join(SUBPROJECTS_DIR, 'CLBlast')
CLBLAST_BUILD_DIR = os.path.join(CLBLAST_DIR, 'build')
CLBLAST_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'CLBlast')

OPENBLAS_DIR = os.path.join(SUBPROJECTS_DIR, 'OpenBLAS')
OPENBLAS_BUILD_DIR = os.path.join(OPENBLAS_DIR, 'build')
OPENBLAS_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'OpenBLAS')
OPENBLAS_NUM_THREADS = 6

BENCHMARK_DIR = os.path.join(SUBPROJECTS_DIR, 'benchmark')
BENCHMARK_BUILD_DIR = os.path.join(BENCHMARK_DIR, 'build')
BENCHMARK_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'benchmark')

GOOGLETEST_DIR = os.path.join(SUBPROJECTS_DIR, 'googletest')
GOOGLETEST_BUILD_DIR = os.path.join(GOOGLETEST_DIR, 'build')
GOOGLETEST_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'googletest')

INIPARSER_DIR = os.path.join(SUBPROJECTS_DIR, 'iniparser')
INIPARSER_BUILD_DIR = os.path.join(INIPARSER_DIR, 'build')
INIPARSER_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'iniparser')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_rebuild', action='store_true')

    return parser.parse_args()


def run_command(args, cwd, exit_on_failure=True):
    cmd = ' '.join(args)

    print('Running command: {}'.format(cmd))

    returncode = subprocess.run(args, cwd=cwd, check=exit_on_failure).returncode

    if exit_on_failure and returncode != 0:
        print('Application {} failed with returncode {}. Exiting...'.format(
            args[0], returncode))
        sys.exit(returncode)

    return returncode


def build_project(force_rebuild, project_dir, project_build_dir, cmake_params):
    if force_rebuild and os.path.exists(project_build_dir):
        shutil.rmtree(project_build_dir)

    if os.path.exists(project_build_dir):
        print('Skip rebuilding \"' + project_dir + '\" use previous build result')
        return

    prepare_command = ['cmake', '-B', project_build_dir, '-DCMAKE_BUILD_TYPE=Release']

    if cmake_params is not None:
        prepare_command = prepare_command + cmake_params

    run_command(prepare_command, project_dir)

    compile_command =  ['cmake' , '--build', project_build_dir, '--config', 'Release', '-j']
    run_command(compile_command, project_dir)


def package_project(force_rebuild, project_build_dir, project_resources_dir, file_names):
    if force_rebuild and os.path.exists(project_resources_dir):
        shutil.rmtree(project_resources_dir)

    os.makedirs(project_resources_dir, exist_ok=True)

    for file_name in file_names:
        shutil.copyfile(os.path.join(project_build_dir, file_name), os.path.join(project_resources_dir, file_name))
    

def main():
    args = get_args()

    os.makedirs(RESOURCES_DIR, exist_ok=True)

    if os.path.exists(CLBLAST_DIR):
        build_project(args.force_rebuild, CLBLAST_DIR, CLBLAST_BUILD_DIR, ['-DCMAKE_CXX_STANDARD=17', '-DCMAKE_POLICY_VERSION_MINIMUM=3.10'])
        package_project(args.force_rebuild, os.path.join(CLBLAST_BUILD_DIR, 'Release'), CLBLAST_RESOURCES_DIR, ['clblast.lib', 'clblast.dll'])

    if os.path.exists(OPENBLAS_DIR):
        build_project(args.force_rebuild, OPENBLAS_DIR, OPENBLAS_BUILD_DIR, ['-DNUM_THREADS=' + str(OPENBLAS_NUM_THREADS), '-DBUILD_TESTING=False', '-DBUILD_BENCHMARKS=False', '-DBUILD_WITHOUT_LAPACK=True', '-DNOFORTRAN=True'])
        package_project(args.force_rebuild, os.path.join(OPENBLAS_BUILD_DIR, 'lib', 'RELEASE'), OPENBLAS_RESOURCES_DIR, ['openblas.lib'])
        package_project(args.force_rebuild, os.path.join(OPENBLAS_BUILD_DIR), OPENBLAS_RESOURCES_DIR, ['config.h', 'openblas_config.h'])
    
    if os.path.exists(BENCHMARK_DIR):
        build_project(args.force_rebuild, BENCHMARK_DIR, BENCHMARK_BUILD_DIR, ['-DBENCHMARK_ENABLE_TESTING=False'])
        package_project(args.force_rebuild, os.path.join(BENCHMARK_BUILD_DIR, 'src', 'Release'), BENCHMARK_RESOURCES_DIR, ['benchmark.lib'])

    if os.path.exists(GOOGLETEST_DIR):
        build_project(args.force_rebuild, GOOGLETEST_DIR, GOOGLETEST_BUILD_DIR, None)
        package_project(args.force_rebuild, os.path.join(GOOGLETEST_BUILD_DIR, 'lib', 'Release'), GOOGLETEST_RESOURCES_DIR, ['gtest.lib', 'gtest_main.lib', 'gmock.lib', 'gmock_main.lib'])

    if os.path.exists(INIPARSER_DIR):
        build_project(args.force_rebuild, INIPARSER_DIR, INIPARSER_BUILD_DIR, ['-DBUILD_DOCS=False', '-DBUILD_EXAMPLES=False', '-DBUILD_TESTING=False'])
        package_project(args.force_rebuild, os.path.join(INIPARSER_BUILD_DIR, 'Release'), INIPARSER_RESOURCES_DIR, ['iniparser.lib', 'libiniparser.lib', 'iniparser.dll'])


if __name__ == '__main__':
    main()
