# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
# 
# SPDX-License-Identifier: Apache-2.0

project('hpcanalysis', 'cpp', version: '0.1.0', default_options: ['cpp_std=c++20'])

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11', required: true)
openmp_dep = dependency('openmp', required: true) 
cpp = meson.get_compiler('cpp')

pkg_source_path = 'src/hpcanalysis'
pkg_name = 'hpcanalysis'
pkg_install_dir = join_paths(py.get_install_dir(), pkg_name)


install_subdir(
    pkg_source_path, 
    install_dir: pkg_install_dir,
    strip_directory: true
)


cpp_sources = files(
    'src/hpcanalysis/cpp/binary_search.cpp',
    'src/hpcanalysis/cpp/parsing.cpp',
    'src/hpcanalysis/cpp/data_read.cpp',
    'src/hpcanalysis/cpp/binding.cpp',
)


py.extension_module(
    'hpcanalysis_cpp', 
    cpp_sources,
    include_directories: include_directories('src/hpcanalysis/cpp'), 
    dependencies: [pybind11_dep, openmp_dep], 
    install: true,
    subdir: pkg_name ,
    cpp_args : ['-std=c++2a'], 
)
