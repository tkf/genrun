import io

import numpy
import pytest

from genrun import dump_any, cli_gen, cli_run, cli_unlock, gen_parameters, \
    cli_progress, print_table, iter_axes, get_axes, make_parser


@pytest.mark.parametrize('src_axes, names', [
    (dict(alpha=[]), ['alpha']),
    (dict(alpha=[], beta=[]), ['alpha', 'beta']),
    ([dict(beta=[]), dict(alpha=[])], ['beta', 'alpha']),
])
def test_iter_axes(src_axes, names):
    actual, _ = zip(*iter_axes(src_axes))
    numpy.testing.assert_equal(actual, names)


@pytest.mark.parametrize('src_axes, axes', [
    (dict(alpha='[0, 1]'), dict(alpha=[0, 1])),
    (dict(alpha='arange(2)'), dict(alpha=[0, 1])),
    (dict(alpha=[0, 1]), dict(alpha=[0, 1])),
    ([dict(alpha=[0, 1])], dict(alpha=[0, 1])),
])
def test_get_axes(src_axes, axes):
    actual = get_axes(dict(axes=src_axes))
    numpy.testing.assert_equal(actual, axes)
    assert actual == axes


@pytest.mark.parametrize('src_axes, parameters', [
    (dict(alpha=[0, 1]), [dict(alpha=0), dict(alpha=1)]),
    (dict(alpha=[0, 1], beta=[10, 11]), [
        dict(alpha=0, beta=10), dict(alpha=0, beta=11),
        dict(alpha=1, beta=10), dict(alpha=1, beta=11),
    ]),
    ([dict(beta=[10, 11]), dict(alpha=[0, 1])], [
        dict(alpha=0, beta=10), dict(alpha=1, beta=10),
        dict(alpha=0, beta=11), dict(alpha=1, beta=11),
    ]),
])
def test_gen_parameters(src_axes, parameters):
    src = dict(base={}, axes=src_axes)
    actual = list(gen_parameters(src, {}))
    numpy.testing.assert_equal(actual, parameters)


def test_gen_parameters_preprocess():
    def preprocess(param):
        if param['x'] + param['y'] > 0:
            return param

    src = dict(
        base={},
        axes=dict(
            x='[-1, 0, 1]',
            y='[-1, 0, 1]',
        )
    )
    runspec = dict(preprocess=preprocess)
    parameters = list(gen_parameters(src, runspec))
    assert parameters == [
        {'x': 0, 'y': 1},
        {'x': 1, 'y': 0},
        {'x': 1, 'y': 1},
    ]


RUNPY = {}
RUNPY["default"] = r"""
def run(dirpath, **_):
    return {
        'command': ['python', '-i', dirpath],
        'input': r'''
import os
import sys
dirpath = sys.argv[-1]
open(os.path.join(dirpath, "argv"), "w").write("\n".join(sys.argv))
open(os.path.join(dirpath, "cwd"), "w").write(os.getcwd())
''',
    }


def is_finished(dirpath, **_):
    import os
    return os.path.exists(os.path.join(dirpath, 'finished'))
"""

RUNPY["noinput"] = """
def run(dirpath, **_):
    return {
        'command': ['python', '-c', r'import os; \
import sys; \
dirpath = sys.argv[-1]; \
open(os.path.join(dirpath, "argv"), "w").write("\\n".join(sys.argv)); \
open(os.path.join(dirpath, "cwd"), "w").write(os.getcwd())', dirpath],
    }
"""


def make_runpy(tmpdir, code=RUNPY["default"], filename="run.py"):
    run_file = tmpdir.join(filename)
    run_file.write(code)
    return str(run_file)


def make_source(tmpdir, axes, base={},
                format="{i}/run.json", filename="source.json"):
    source_file = str(tmpdir.join(filename))
    dump_any(source_file, dict(
        base=base,
        axes=axes,
        format=format,
    ))
    return source_file


@pytest.mark.parametrize('num', [1, 3])
def test_gen(tmpdir, num):
    source_file = make_source(tmpdir, axes={'alpha': repr(range(num))})
    run_file = '/dev/null'
    cli_gen(source_file, run_file)

    dirs = tmpdir.listdir(lambda p: p.check(dir=True))
    assert len(dirs) == num
    exists = [d.join("run.json").check() for d in dirs]
    assert all(exists)


@pytest.mark.parametrize('num', [1, 3])
@pytest.mark.parametrize('runpy', ['default', 'noinput'])
def test_run(tmpdir, num, runpy):
    source_file = make_source(tmpdir, axes={'alpha': repr(range(num))})
    run_file = make_runpy(tmpdir, RUNPY[runpy])
    cli_gen(source_file, run_file)
    cli_run(source_file, run_file, None, None)

    dirs = tmpdir.listdir(lambda p: p.check(dir=True))
    assert len(dirs) == num
    assert all(d.join("argv").check() for d in dirs)
    assert all(d.join("cwd").check() for d in dirs)
    assert [d.join("argv").readlines()[-1] for d in dirs] == dirs
    assert [d.join("cwd").read() for d in dirs] == dirs


def test_lock(tmpdir):
    """
    Locked directory (i.e., with .lock file) should not be executed.
    """
    source_file = make_source(tmpdir, axes={'alpha': 'range(2)'})
    run_file = make_runpy(tmpdir)
    cli_gen(source_file, run_file)
    tmpdir.ensure("0", ".lock")
    cli_run(source_file, run_file, None, None)

    d0, d1 = sorted(tmpdir.listdir(lambda p: p.check(dir=True)))
    assert not d0.join("argv").check()
    assert d1.join("argv").check()


def test_unlock(tmpdir):
    source_file = make_source(tmpdir, axes={'alpha': 'range(2)'})
    run_file = make_runpy(tmpdir)
    cli_gen(source_file, run_file)
    assert all(tmpdir.join(d, 'run.json').check() for d in '01')
    tmpdir.ensure("0", ".lock")
    tmpdir.ensure("1", ".lock")
    tmpdir.ensure("0", "some_result")

    cli_unlock(source_file, run_file)
    assert tmpdir.join("0", ".lock").check()
    assert not tmpdir.join("1", ".lock").check()


@pytest.mark.parametrize('lockall', [False, True])
def test_progress(tmpdir, capsys, lockall):
    source_file = make_source(tmpdir, axes={
        'alpha': 'range(3)',
        'beta': 'range(2)',
        'gamma': 'range(2)',
    })
    run_file = make_runpy(tmpdir)
    cli_gen(source_file, run_file)
    assert all(tmpdir.join(str(d), 'run.json').check() for d in range(12))
    for d in range(7):
        tmpdir.ensure(str(d), ".lock")
        tmpdir.ensure(str(d), "finished")
    if lockall:
        for d in range(12):
            tmpdir.ensure(str(d), ".lock")

    out0, err0 = capsys.readouterr()
    assert out0 == err0 == ''

    stream = io.StringIO()
    print_table([
        ['', '0', '1'],
        ['0', 'OK', 'OK'],
        ['1', 'OK', '50'],
        ['2', '0', '0'],
    ], file=stream)
    table = stream.getvalue()

    cli_progress(source_file, run_file, False)
    out, err = capsys.readouterr()
    assert table in out
    assert err == ''


def get_subcommands():
    parser = make_parser()
    action = parser._subparsers._group_actions[0]
    return action.choices.keys()


@pytest.mark.parametrize('sub_command', [None] + list(get_subcommands()))
def test_argparse_help(sub_command):
    parser = make_parser()
    args = ['--help'] if sub_command is None else [sub_command, '--help']
    with pytest.raises(SystemExit) as errinfo:
        parser.parse_args(args)
    assert errinfo.value.code == 0
