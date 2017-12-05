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

import collections
import copy
import functools
import itertools
import logging
import os
import subprocess
import sys

__version__ = '0.0.0'
__author__ = 'Takafumi Arakaki'
__license__ = None


logger = logging.getLogger("genrun")


class GenRunExit(RuntimeError):
    """ Error to be raised on known erroneous situations. """


def coroutine_send(func):
    @functools.wraps(func)
    def start(*args, **kwds):
        cr = func(*args, **kwds)
        next(cr)
        return cr.send
    return start


class JsonModule(object):

    def __init__(self, json):
        self.json = json
        self.load = json.load

    def dump(self, obj, file):
        self.json.dump(obj, file, sort_keys=True)


class NDJsonModule(object):

    def __init__(self, json):
        self.json = json

    def load(self, file):
        raise NotImplementedError

    def dump(self, obj, file):
        json = self.json
        for d in obj:
            json.dump(d, file)
            file.write('\n')


def param_module(path):
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml
    elif path.lower().endswith(".json"):
        import json
        return JsonModule(json)
    elif path.lower().endswith(".ndjson"):
        import json
        return NDJsonModule(json)
    else:
        raise ValueError('data format of {!r} is not supported'.format(path))


def load_any(path):
    """
    Load data from given path; data format is determined by file extension
    """
    loader = param_module(path).load
    with open(path) as f:
        return loader(f)


def dump_any(dest, obj, filetype=None):
    if filetype is None:
        filetype = dest
    else:
        # Prepend '.' since param_module dispatches extension...
        filetype = '.' + filetype  # TODO: refactoring

    dumper = param_module(filetype).dump
    if hasattr(dest, 'write'):
        dumper(obj, dest)
    else:
        with open(dest, 'w') as f:
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


def has_common_keys(dicts):
    if not dicts:
        return False
    for d in dicts[1:]:
        if set(dicts[0]) & set(d):
            return True
    return has_common_keys(dicts[1:])


def iter_axes_list(src_axes):
    if not all(isinstance(sub, dict) for sub in src_axes):
        raise GenRunExit(
            "list type 'axes' in source file must contain only dicts;"
            " got: {}"
            .format(list(map(type, src_axes))))
    if has_common_keys(src_axes):
        raise GenRunExit(
            "list type 'axes' in source file must not contain dicts"
            " with common keys")
    for sub in src_axes:
        for kv in iter_axes(sub):
            yield kv


def iter_axes(src_axes):
    if not src_axes:
        raise GenRunExit("Empty 'axes' in source file.")
    if isinstance(src_axes, dict):
        return sorted(src_axes.items(), key=lambda x: x[0])
    if isinstance(src_axes, list):
        return iter_axes_list(src_axes)
    raise GenRunExit("'axes' in source file cannot be of type {}"
                     .format(type(src_axes)))


def get_axes(src, debug=False):
    axes = collections.OrderedDict()
    for name, code in iter_axes(src['axes']):
        if isinstance(code, list):
            axes[name] = code
            continue
        try:
            axes[name] = list(src_eval(code))
        except:
            if not debug:
                raise
            import pdb
            pdb.post_mortem()
    return axes


def prepare_gen(src, runspec, debug=False):
    axes = get_axes(src, debug)
    preprocess = runspec.get('preprocess', lambda x: x)
    keys = list(axes.keys())
    return axes, preprocess, keys


def unprocessed_parameters(base, keys, parameters):
    for vals in parameters:
        param = copy.deepcopy(base)
        for k, v in zip(keys, vals):
            set_dotted(param, k, v)
        yield param


def gen_parameters(src, runspec, debug=False):
    axes, preprocess, keys = prepare_gen(src, runspec, debug)
    parameters = itertools.product(*[axes[name] for name in keys])
    unprocessed = unprocessed_parameters(src['base'], keys, parameters)
    return filter(None, map(preprocess, unprocessed))


def param_path(src, basedir, i):
    return os.path.join(basedir, src['format'].format(**locals()))


def is_unstarted(filepath):
    dirpath = os.path.dirname(filepath)
    return set(os.listdir(dirpath)) == {'.lock', os.path.basename(filepath)}


def two_step_generator(src, runspec, debug, num_head=2):
    axes, preprocess, keys = prepare_gen(src, runspec, debug)
    focused_keys = keys[:num_head]
    rest_keys = keys[num_head:]

    unprocessed0 = unprocessed_parameters(
        src['base'], focused_keys,
        itertools.product(*[axes[name] for name in focused_keys]))
    focused_ids = itertools.product(*[range(len(axes[name]))
                                      for name in focused_keys])
    focused_axes = {name: axes[name] for name in focused_keys}

    def outer_iterator():
        from argparse import Namespace
        ns = Namespace()
        ns.i = 0
        for focus, param0 in zip(focused_ids, unprocessed0):
            def iterator():
                ps = itertools.product(*[axes[name] for name in rest_keys])
                for param in unprocessed_parameters(param0, rest_keys, ps):
                    param = preprocess(param)
                    if param is not None:
                        yield ns.i, param
                        ns.i += 1
            yield focus, iterator()

    return focused_keys, focused_axes, outer_iterator()


def analyze_progress(src, runspec, debug, source_file):
    basedir = os.path.dirname(source_file)
    is_finished = runspec['is_finished']

    focused_keys, focused_axes, outer_iterator = \
        two_step_generator(src, runspec, debug)

    progress = {}
    for focus, iterator in outer_iterator:
        count = collections.Counter()
        for i, param in iterator:
            filepath = param_path(src, basedir, i)
            if is_unstarted(filepath):
                count['unstarted'] += 1
            elif is_finished(
                    param=param,
                    dirpath=os.path.dirname(filepath),
                    filename=os.path.basename(filepath),
                    filepath=filepath,
                    source=src,
                    ):
                count['finished'] += 1
            else:
                count['running'] += 1
        progress[focus] = count

    return focused_keys, focused_axes, progress


def progress_to_table(focused_keys, focused_axes, progress):
    xk, yk = focused_keys
    table = [[''] + list(map(str, range(len(focused_axes[xk]))))]
    for j in range(len(focused_axes[yk])):
        row = [str(j)]
        for i in range(len(focused_axes[xk])):
            try:
                count = progress[i, j]
                assert len(count) > 0
            except (KeyError, AssertionError):
                row.append('-')
                continue

            ratio = count.get('finished', 0) / sum(count.values())
            if ratio == 1:
                row.append('OK')
            else:
                row.append(str(int(ratio * 100)))
        table.append(row)
    return table


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


def print_table(table, sep='\t', **print_kwargs):
    table = [list(map(str, row)) for row in table]
    widths = list(map(max, zip(*(map(len, row) for row in table))))
    for row in table:
        print(*[s.ljust(w) for s, w in zip(row, widths)], sep=sep,
              **print_kwargs)


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
        logger.info('Using: %s', source_file)
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
        logger.info('Using: %s', run_file)
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


@coroutine_send
def oneline_reporter(header):
    message = yield
    if message is None:
        yield
        return
    print(header, end=' ')
    while True:
        print(message, end=' ', flush=True)
        message = yield
        if message is None:
            break
    print()
    yield


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

    report_skip = oneline_reporter('Skipping locked directories:')
    for path in param_files:
        path = os.path.abspath(path)

        lock_file = os.path.join(os.path.dirname(path), '.lock')
        if os.path.exists(lock_file):
            report_skip(os.path.dirname(lock_file))
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
    report_skip(None)


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
    report_skip = oneline_reporter('Skipping locked directories:')
    for i, _ in enumerate(gen_parameters(src, runspec)):
        path = param_path(src, basedir, i)
        if path in remaining:
            remaining.remove(path)
            lock_file = os.path.join(os.path.dirname(path), '.lock')
            if os.path.exists(lock_file):
                report_skip(os.path.dirname(lock_file))
                continue
            ids.append(i)
            lock_file_list.append(lock_file)
            open(lock_file, 'w').close()
    report_skip(None)
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
        if is_unstarted(filepath):
            yield filepath


def cli_unlock(source_file, run_file):
    """
    Remove .lock files from unfinished run directories.

    The run directories with only the .lock and parameter files are
    considered unfinished.  To list directories to be unlocked (i.e.,
    dry-run), use list-unfinished command.

    """
    report_remove = oneline_reporter('Removing:')
    for filepath in find_unfinished(source_file, run_file):
        lockfile = os.path.join(os.path.dirname(filepath), '.lock')
        report_remove(lockfile)
        os.remove(lockfile)
    report_remove(None)


def cli_list_unfinished(source_file, run_file):
    """
    List unfinished run directories.

    See also: unlock
    """
    for filepath in find_unfinished(source_file, run_file):
        print(os.path.dirname(filepath))


def cli_progress(source_file, run_file, debug):
    """
    [EXPERIMENTAL] Show %finish of two outer-most axes.

    Note that this command works only for run with two or more source
    axes at the moment.

    """
    import numpy

    source_file = find_source_file(source_file)
    run_file = find_run_file(run_file)
    src = load_any(source_file)
    runspec = load_run(run_file)
    focused_keys, focused_axes, progress = \
        analyze_progress(src, runspec, debug, source_file)

    table = progress_to_table(focused_keys, focused_axes, progress)
    xk, yk = focused_keys
    if len(focused_axes[xk]) > len(focused_axes[yk]):
        xk, yk = yk, xk
        table = list(zip(*table))

    print('%finish of two outer-most axes:', *focused_keys)
    print()
    print_table(table)

    print()
    print('X-axis:', xk)
    print(numpy.array(focused_axes[xk]))
    print('Y-axis:', yk)
    print(numpy.array(focused_axes[yk]))


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


def cli_axes_keys(source_file, delimiter, end, debug):
    """
    Print axes keys in the order defined in `source_file`.
    """
    source_file = find_source_file(source_file)
    src = load_any(source_file)
    axes = get_axes(src, debug=debug)
    print(*axes.keys(), sep=delimiter, end=end)


def load_source(source_file, debug, deaxes):
    source_file = find_source_file(source_file)
    src = load_any(source_file)
    if deaxes:
        axes = get_axes(src, debug=debug)
        return dict(src['base'], **axes)
    else:
        return src


def cli_cat(source_files, output_type, debug, deaxes, output,
            path_key):
    """
    Load `source_files`, concatenate them.
    """
    sources = [load_source(f, debug, deaxes) for f in source_files]
    if path_key:
        sources = [dict(d, **{path_key: f}) for d, f
                   in zip(sources, source_files)]

    dump_any(sys.stdout if output == '-' else output,
             sources,
             output_type)


def make_parser(doc=__doc__):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=doc)
    subparsers = parser.add_subparsers()

    def subp(command, func):
        doc = (func.__doc__ or '').replace('%', '%%')
        title = None
        for title in filter(None, map(str.strip, doc.splitlines())):
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

    p = subp('progress', cli_progress)
    p.add_argument('--debug', action='store_true')
    add_argument_source_file(p)
    add_argument_run_file(p)

    p = subp('len', cli_len)
    add_argument_source_file(p)
    add_argument_run_file(p)

    p = subp('axes-keys', cli_axes_keys)
    add_argument_source_file(p)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--delimiter', default='\n')
    p.add_argument('--end', default='\n')

    p = subp('cat', cli_cat)
    p.add_argument(
        'source_files', metavar='source_file', nargs='+',
        help='Path to parameter configuration files.')
    p.add_argument('--debug', action='store_true')
    p.add_argument(
        '--output-type', default='ndjson',
        choices=('ndjson', 'json', 'yaml'))
    p.add_argument(
        '--output', default='-',
        help='Path to which output is written. "-" means stdout.')
    p.add_argument(
        '--deaxes', action='store_true',
        help='Mix "axes" and "base" in `source_file`.')
    p.add_argument(
        '--path-key',
        help='Include `source_file` with this key in each record.')

    return parser


def main(args=None):
    parser = make_parser()
    ns = parser.parse_args(args)

    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s')

    if not hasattr(ns, 'func'):
        parser.print_usage()
        parser.exit(2)
    try:
        return (lambda func, **kwds: func(**kwds))(**vars(ns))
    except GenRunExit as err:
        parser.exit(1, str(err).strip() + '\n')


if __name__ == '__main__':
    main()
