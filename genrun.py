#!/usr/bin/env python3

# [[[cog import cog; cog.outl('"""\n%s\n"""' % file('README.rst').read())]]]
"""
=============================================================
 genrun -- generate parameter files and run programs on them
=============================================================


Examples
========
Simply running::

   genrun all BASE/source.yaml PATH/TO/run.py

where `BASE/source.yaml` is

.. code:: yaml

   base:
       spam:
           egg:
               omega: 0
   axes:
       spam.egg.alpha: 'arange(0.0, 2.0, 0.1)'
       spam.egg.beta: 'linspace(-1, 1, num=10)'
       spam.gamma: 'logspace(0, 3, base=2, num=10)'
   format: '{i}/run.json'

does what conceptually equivalently to the following code

.. code:: python

   for numpy import arange, linspace, logspace

   for alpha, beta, gamma in product(arange(0.0, 2.0, 0.1),
                                     linspace(-1, 1, num=10),
                                     logspace(0, 3, base=2, num=10)):
       param = base.copy()
       param['spam']['egg']['alpha'] = alpha
       param['spam']['egg']['beta'] = beta
       param['spam']['gamma'] = gamma
       run(param)

where the function `run` saves `param` and does what is defined in
``run.py`` (e.g., submit a job via ``qsub``; see blow).


Running ``my_script`` via ``qsub``
----------------------------------

.. code:: python

   def run(param, filename, dirpath, **_):
       return {
           'command': 'qsub -o stdout.log -e stderr.log',
           'input': '''
           cd '{dirpath}'
           my_script '{filename}'
           '''.format(**locals()),
       }

"""
# [[[end]]]

import copy
import os
import itertools
import subprocess

__version__ = '0.0.0'
__author__ = 'Takafumi Arakaki'
__license__ = None


class GenRunExit(RuntimeError):
    """ Error to be raised on known erroneous situations. """


class JsonModule(object):

    def __init__(self, json):
        self.json = json
        self.load = json.load

    def dump(self, obj, file):
        self.json.dump(obj, file, sort_keys=True)


def param_module(path):
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml
    elif path.lower().endswith(".json"):
        import json
        return JsonModule(json)
    else:
        raise ValueError('data format of {!r} is not supported'.format(path))


def load_any(path):
    """
    Load data from given path; data format is determined by file extension
    """
    loader = param_module(path).load
    with open(path) as f:
        return loader(f)


def dump_any(path, obj):
    dumper = param_module(path).dump
    with open(path, 'w') as f:
        dumper(obj, f)


def src_eval(code):
    import numpy
    return eval(code, None, vars(numpy))


def set_dotted(d, k, v):
    parts = k.split('.')
    p = d
    for i in parts[:-1]:
        p = p.setdefault(i, {})
    p[parts[-1]] = v


def gen_parameters(src, debug=False):
    axes = {}
    for name, code in sorted(src['axes'].items(), key=lambda x: x[0]):
        try:
            axes[name] = list(src_eval(code))
        except:
            if not debug:
                raise
            import pdb
            pdb.post_mortem()

    keys = sorted(axes)
    parameters = itertools.product(*[axes[name] for name in keys])
    for i, vals in enumerate(parameters):
        param = copy.deepcopy(src['base'])
        for k, v in zip(keys, vals):
            set_dotted(param, k, v)
        yield param


def param_path(src, basedir, i):
    return os.path.join(basedir, src['format'].format(**locals()))


def load_run(run_file):
    with open(run_file) as f:
        code = f.read()
    ns = {}
    exec(code, None, ns)
    return ns


SOURCE_FILE_CANDIDATES = ['source.yaml', 'source.json']


def find_source_file(source_file):
    if not source_file:
        for source_file in SOURCE_FILE_CANDIDATES:
            if os.path.exists(source_file):
                break
        else:
            raise GenRunExit('source_file is not given and none of the'
                             ' following files are present in the current'
                             ' directory: {}'
                             .format(', '.join(SOURCE_FILE_CANDIDATES)))
        print('Using:', source_file)
    return source_file


def find_run_file(run_file):
    if not run_file:
        directory = os.getcwd()
        parent = os.path.dirname(directory)
        while parent != directory:
            run_file = os.path.join(directory, 'run.py')
            if os.path.exists(run_file):
                break
            directory = parent
            parent = os.path.dirname(parent)
        else:
            raise GenRunExit('run_file is not given and run.py is'
                             ' not found in here or any of the parent'
                             ' directories.')
        print('Using:', run_file)
    return run_file


def cli_gen(source_file, debug=False):
    """
    Generate parameter files based on `source_file`.
    """
    source_file = find_source_file(source_file)
    src = load_any(source_file)

    basedir = os.path.dirname(source_file)
    for i, param in enumerate(gen_parameters(src, debug)):
        filepath = param_path(src, basedir, i)
        os.makedirs(os.path.dirname(filepath))
        dump_any(filepath, param)


def cli_run(source_file, run_file, param_files):
    """
    Run generated parameter files.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)

    if not param_files:
        basedir = os.path.dirname(source_file)
        nparam = sum(1 for _ in gen_parameters(src))  # FIXME: optimize
        param_files = [param_path(src, basedir, i) for i in range(nparam)]

    runspec = load_run(run_file)
    for path in param_files:
        path = os.path.abspath(path)

        lock_file = os.path.join(os.path.dirname(path), '.lock')
        if os.path.exists(lock_file):
            print(lock_file, 'exists. skipping...')
            continue
        open(lock_file, 'w').close()

        param = load_any(path)
        cmdspec = runspec['run'](
            param=param,
            dirpath=os.path.dirname(path),
            filename=os.path.basename(path),
            filepath=path,
            source=src,
        )
        command = cmdspec.pop('command')
        pinput = cmdspec.pop('input', None)
        proc = subprocess.Popen(
            command,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            **dict(
                shell=isinstance(command, str),
                cwd=os.path.dirname(path),
                **cmdspec)
        )
        proc.communicate(pinput)
        if proc.returncode != 0:
            os.remove(lock_file)
            raise RuntimeError("{} failed".format(command))


def cli_all(source_file, run_file):
    """
    Generate parameter files and then run them.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    cli_gen(source_file)
    cli_run(source_file, run_file, param_files=None)


def make_parser(doc=__doc__):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=doc)
    subparsers = parser.add_subparsers()

    def subp(command, func):
        doc = func.__doc__
        title = None
        for title in filter(None, map(str.strip, (doc or '').splitlines())):
            break
        p = subparsers.add_parser(
            command,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            help=title,
            description=doc)
        p.set_defaults(func=func)
        return p

    def add_argument_source_file(p):
        p.add_argument('source_file', nargs='?', help="""
        Path to parameter configuration file.  If not given or an
        empty string, searched from the following files (in this
        order): {}
        """.format(', '.join(SOURCE_FILE_CANDIDATES)))

    def add_argument_run_file(p):
        p.add_argument('run_file', nargs='?', help="""
        Path to run configuration file (typically run.py).  If not
        given or an empty string, run.py is searched from current
        directory or parent directories.
        """)

    p = subp('gen', cli_gen)
    p.add_argument('--debug', action='store_true')
    add_argument_source_file(p)

    p = subp('run', cli_run)
    add_argument_source_file(p)
    add_argument_run_file(p)
    p.add_argument('param_files', nargs='*')

    p = subp('all', cli_all)
    add_argument_source_file(p)
    add_argument_run_file(p)

    return parser


def main(args=None):
    parser = make_parser()
    ns = parser.parse_args(args)
    if not hasattr(ns, 'func'):
        parser.print_usage()
        parser.exit(2)
    try:
        return (lambda func, **kwds: func(**kwds))(**vars(ns))
    except GenRunExit as err:
        parser.exit(1, str(err).strip() + '\n')


if __name__ == '__main__':
    main()
