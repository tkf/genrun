#!/usr/bin/env python3

# [[[cog import cog; cog.outl('"""\n%s\n"""' % file('README.rst').read())]]]
"""
=============================================================
 genrun -- generate parameter files and run programs on them
=============================================================


Examples
========
Simply running::

   genrun all

where `./source.yaml` is

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

from __future__ import print_function

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


def gen_parameters(src, runspec, debug=False):
    axes = {}
    for name, code in sorted(src['axes'].items(), key=lambda x: x[0]):
        try:
            axes[name] = list(src_eval(code))
        except:
            if not debug:
                raise
            import pdb
            pdb.post_mortem()

    try:
        preprocess = runspec['preprocess']
    except KeyError:
        def preprocess(param):
            return param

    keys = sorted(axes)
    parameters = itertools.product(*[axes[name] for name in keys])
    for i, vals in enumerate(parameters):
        param = copy.deepcopy(src['base'])
        for k, v in zip(keys, vals):
            set_dotted(param, k, v)
        param = preprocess(param)
        if param is not None:
            yield param


def param_path(src, basedir, i):
    return os.path.join(basedir, src['format'].format(**locals()))


def load_run(run_file):
    with open(run_file) as f:
        code = f.read()
    ns = {}
    exec(code, ns)
    return ns


def indices_to_range(indices):
    """
    Convert a list of integers to the range format for sbatch and qsub.

    >>> indices_to_range([0, 1, 2, 3])
    '0-3'
    >>> indices_to_range([0, 10, 11, 12])
    '0,10-12'
    >>> indices_to_range([0, 1, 2, 10, 11, 12, 100, 101, 102])
    '0-2,10-12,100-102'

    See:

    - https://slurm.schedmd.com/sbatch.html
    - http://docs.adaptivecomputing.com/torque/4-2-7/help.htm#topics/commands/qsub.htm#-t

    """
    assert len(indices) == len(set(indices))

    ranges = []
    it = iter(indices)
    try:
        i = next(it)
    except StopIteration:
        return ''

    nonempty = True
    while nonempty:
        prev = beg = i
        for i in it:
            if i != prev + 1:
                break
            prev = i
        else:
            nonempty = False

        if beg == prev:
            ranges.append(str(beg))
        else:
            ranges.append('{}-{}'.format(beg, prev))

    return ','.join(ranges)


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


def cli_gen(source_file, run_file, debug=False):
    """
    Generate parameter files based on `source_file`.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)

    basedir = os.path.dirname(source_file)
    for i, param in enumerate(gen_parameters(src, runspec, debug)):
        filepath = param_path(src, basedir, i)
        os.makedirs(os.path.dirname(filepath))
        dump_any(filepath, param)


def run_loop(source_file, run_file, param_files):
    """
    Run generated parameter files.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)

    if not param_files:
        basedir = os.path.dirname(source_file)
        nparam = sum(1 for _ in gen_parameters(src, runspec))  # FIXME:optimize
        param_files = [param_path(src, basedir, i) for i in range(nparam)]

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


def run_array(source_file, run_file, param_files):
    """
    Run generated parameter files.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)

    if not param_files:
        basedir = os.path.dirname(source_file)
        nparam = sum(1 for _ in gen_parameters(src, runspec))  # FIXME:optimize
        param_files = [param_path(src, basedir, i) for i in range(nparam)]

    ids = []
    lock_file_list = []
    remaining = set(param_files)
    for i, _ in enumerate(gen_parameters(src, runspec)):
        path = param_path(src, basedir, i)
        if path in remaining:
            remaining.remove(path)
            lock_file = os.path.join(os.path.dirname(path), '.lock')
            if os.path.exists(lock_file):
                print(lock_file, 'exists. skipping...')
                continue
            ids.append(i)
            lock_file_list.append(lock_file)
            open(lock_file, 'w').close()
    assert not remaining

    cmdspec = runspec['run_array'](
        ids=ids,
        array=indices_to_range(ids),
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
            **cmdspec)
    )
    proc.communicate(pinput)
    if proc.returncode != 0:
        print('Rolling back (removing lock files)')
        for lock_file in lock_file_list:
            print('.', end='', flush=True)
            try:
                os.remove(lock_file)
            except OSError:
                print('! Cannot remove', lock_file)
        print()
        raise GenRunExit('{} failed'.format(command))


def cli_all(source_file, run_file, run_type):
    """
    Generate parameter files and then run them.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    cli_gen(source_file, run_file)
    cli_run(source_file, run_file, param_files=None, run_type=run_type)


def cli_run(source_file, run_file, param_files, run_type):
    """
    Run generated parameter files.
    """
    run_file = find_run_file(run_file)
    runspec = load_run(run_file)
    if {'run', 'run_array'} <= set(runspec):
        if not run_type:
            raise GenRunExit(
                'Ambiguous `run_type`.  Run file {} contains both'
                ' `run` and `run_array` functions while --use-array'
                ' nor --use-loop is given.'
                .format(run_file))
    else:
        run_type = 'array' if 'run_array' in runspec else 'loop'

    if run_type == 'array':
        run_array(source_file, run_file, param_files)
    else:
        run_loop(source_file, run_file, param_files)


def find_unfinished(source_file, run_file):
    """
    Return an iterator yielding unfinished parameter files.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)
    basedir = os.path.dirname(source_file)
    for i, param in enumerate(gen_parameters(src, runspec)):
        filepath = param_path(src, basedir, i)
        dirpath = os.path.dirname(filepath)
        if set(os.listdir(dirpath)) == {'.lock', os.path.basename(filepath)}:
            yield filepath


def cli_unlock(source_file, run_file):
    """
    Remove .lock files from unfinished run directories.

    The run directories with only the .lock and parameter files are
    considered unfinished.  To list directories to be unlocked (i.e.,
    dry-run), use list-unfinished command.

    """
    for filepath in find_unfinished(source_file, run_file):
        lockfile = os.path.join(os.path.dirname(filepath), '.lock')
        print('Remove', lockfile)
        os.remove(lockfile)


def cli_list_unfinished(source_file, run_file):
    """
    List unfinished run directories.

    See also: unlock
    """
    for filepath in find_unfinished(source_file, run_file):
        print(os.path.dirname(filepath))


def cli_len(source_file, run_file):
    """
    Print number of parameter files to be generated.
    """
    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)
    nparam = sum(1 for _ in gen_parameters(src, runspec))
    print(nparam)


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
        p.add_argument('--source-file', help="""
        Path to parameter configuration file.  If not given or an
        empty string, searched from the following files (in this
        order): {}
        """.format(', '.join(SOURCE_FILE_CANDIDATES)))

    def add_argument_run_file(p):
        p.add_argument('--run-file', help="""
        Path to run configuration file.  If not given or an empty
        string, run.py is searched from current directory or parent
        directories.
        """)

    def add_argument_run_type(p):
        p.add_argument('--use-array', dest='run_type', default=None,
                       action='store_const', const='array',
                       help="""
        In case run file contains both `run` and `run_array` function,
        indicate that `run_array` must be used.
        """)
        p.add_argument('--use-loop', dest='run_type', default=None,
                       action='store_const', const='loop',
                       help="""
        Similar to --use-array but use `run` function.
        """)

    p = subp('gen', cli_gen)
    p.add_argument('--debug', action='store_true')
    add_argument_source_file(p)
    add_argument_run_file(p)

    p = subp('run', cli_run)
    add_argument_source_file(p)
    add_argument_run_file(p)
    add_argument_run_type(p)
    p.add_argument('param_files', nargs='*')

    p = subp('all', cli_all)
    add_argument_source_file(p)
    add_argument_run_file(p)
    add_argument_run_type(p)

    p = subp('unlock', cli_unlock)
    add_argument_source_file(p)
    add_argument_run_file(p)

    p = subp('list-unfinished', cli_list_unfinished)
    add_argument_source_file(p)
    add_argument_run_file(p)

    p = subp('len', cli_len)
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
