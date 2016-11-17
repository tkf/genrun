#!/usr/bin/env python3

import copy
import os
import itertools
import subprocess

__version__ = '0.0.0'
__author__ = 'Takafumi Arakaki'
__license__ = None


def load_any(path):
    """
    Load any data file from given path (distinguished by file extension)
    """
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        loader = yaml.load
    elif path.lower().endswith(".json"):
        import json
        loader = json.load
    else:
        raise ValueError('data format of {!r} is not supported'.format(path))

    with open(path) as f:
        return loader(f)


def dump_any(path, obj):
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        dumper = yaml.dump
    elif path.lower().endswith(".json"):
        import json
        dumper = json.dump
    else:
        raise ValueError('data format of {!r} is not supported'.format(path))

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
    for name, code in src['axes'].items():
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


def cli_gen(source_file, debug=False):
    """
    Generate parameter files based on `source_file`.
    """
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
    src = load_any(source_file)

    if not param_files:
        basedir = os.path.dirname(source_file)
        nparam = sum(1 for _ in gen_parameters(src))  # FIXME: optimize
        param_files = [param_path(src, basedir, i) for i in range(nparam)]

    runspec = load_run(run_file)
    for path in param_files:

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
        command = cmdspec['command']
        proc = subprocess.run(
            command,
            input=cmdspec['stdin'],
            universal_newlines=True,
            shell=isinstance(command, str),
        )
        if proc.returncode != 0:
            os.remove(lock_file)
            raise RuntimeError("{} failed".format(command))


def cli_all(source_file, run_file):
    """
    Generate parameter files and then run them.
    """
    cli_gen(source_file)
    cli_run(source_file, run_file, param_files=None)


def make_parser(doc=__doc__):
    import argparse

    class FormatterClass(argparse.RawDescriptionHelpFormatter,
                         argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=FormatterClass,
        description=doc)
    subparsers = parser.add_subparsers()

    def subp(command, func):
        doc = func.__doc__
        title = None
        for title in filter(None, map(str.strip, (doc or '').splitlines())):
            break
        p = subparsers.add_parser(
            command,
            formatter_class=FormatterClass,
            help=title,
            description=doc)
        p.set_defaults(func=func)
        return p

    def add_argument_source_file(p):
        p.add_argument('source_file', help="""
        path to parameter configuration file (typically source.yaml).
        """)

    def add_argument_run_file(p):
        p.add_argument('run_file', help="""
        path to run configuration file (typically run.py).
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
    return (lambda func, **kwds: func(**kwds))(**vars(ns))


if __name__ == '__main__':
    main()
