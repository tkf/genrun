import pytest

from genrun import dump_any, cli_gen, cli_run


RUNPY_DEFAULT = r"""
def run(dirpath, **_):
    return {
        'command': ['python', '-i', dirpath],
        'input': r'''
import os
import sys
dirpath = sys.argv[-1]
open(os.path.join(dirpath, "argv"), "w").write('\n'.join(sys.argv))
open(os.path.join(dirpath, "cwd"), "w").write(os.getcwd())
''',
    }
"""


def make_runpy(tmpdir, code=RUNPY_DEFAULT, filename="run.py"):
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
    cli_gen(source_file)

    dirs = tmpdir.listdir(lambda p: p.check(dir=True))
    assert len(dirs) == num
    exists = [d.join("run.json").check() for d in dirs]
    assert all(exists)


@pytest.mark.parametrize('num', [1, 3])
def test_run(tmpdir, num):
    source_file = make_source(tmpdir, axes={'alpha': repr(range(num))})
    run_file = make_runpy(tmpdir)
    cli_gen(source_file)
    cli_run(source_file, run_file, None)

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
    cli_gen(source_file)
    tmpdir.ensure("0", ".lock")
    cli_run(source_file, run_file, None)

    d0, d1 = sorted(tmpdir.listdir(lambda p: p.check(dir=True)))
    assert not d0.join("argv").check()
    assert d1.join("argv").check()
