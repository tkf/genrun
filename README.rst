========
 genrun
========

:Subtitle: parameter search generator
:Manual section: 1
:Manual group: genrun manual


Synopsis
========
::

   genrun all
   genrun gen
   genrun run [param_files [param_files ...]]
   genrun unlock
   genrun list-unfinished
   genrun cat source_file [source_file ...]
   genrun help


Description
===========

`genrun all` is a command to generate parameter files and run a
specified program for each of them.  The first and second steps can be
run separately using `genrun gen` and `genrun run`, respectively.


Examples
========

Create `source.yaml` file

.. code:: yaml

   base:
       spam:
           egg:
               omega: 0
   axes:
       spam.egg.alpha: 'arange(0.0, 2.0, 0.1)'
       spam.egg.beta: 'linspace(-1, 1, num=10)'
       spam.gamma: 'logspace(0, 3, base=2, num=10)'
   format: 'tasks/{i}/run.json'

and `run.py` file

.. code:: python

   def run(**_):
       return {"command": ["my_program", "run.json"]}

Then running `genrun all` is equivalent to the following code

.. code:: python

   for (i, (alpha, beta, gamma)) in enumerate(
       itertools.product(
           numpy.arange(0.0, 2.0, 0.1),
           numpy.linspace(-1, 1, num=10),
           numpy.logspace(0, 3, base=2, num=10),
       )
   ):
       param = {"spam": {"egg": {"omega": 0}}}
       param["spam"]["egg"]["alpha"] = alpha
       param["spam"]["egg"]["beta"] = beta
       param["spam"]["gamma"] = gamma

       with open(f"tasks/{i}/run.json", "w") as file:
           json.dump(param, file)

       subprocess.call(["my_program", "run.json"], cwd=f"tasks/{i}/")

This runs program sequentially.  To gain parallelism, a batch system
like `Task Spooler`_ (``tsp``) can be used:

.. _`Task Spooler`: http://vicerveza.homeunix.net/~viric/soft/ts/

.. code:: python

   def run(**_):
       return {"command": ["tsp", "my_program", "run.json"]}

Note that ``tsp`` is in front of ``my_program`` now.

Using Slurm to run the program
------------------------------

Cluster management system such as Slurm and PBS may prefer to manage
related computing tasks in a "job array" style.  This can be done in
`genrun` by defining ``run_array`` instead of ``run`` in `run.py`
file:

.. code:: python

   script_template = """\
   #!/bin/bash
   #SBATCH --output tasks/%a/stdout.log
   #SBATCH --array {array}

   cd "tasks/$SLURM_ARRAY_TASK_ID"
   srun my_program run.json
   """


   def run_array(**kwds):
       return {
           "command": "sbatch",
           "input": script_template.format(**kwds),
       }


Manual
======

Source parameter file (`source.{toml,yaml,json}`)
-------------------------------------------------

``base``: `dict`
    The base parameter.  This is the non-varying part of the full
    parameter.  The parameters specified by ``axes`` are mixed into
    this ``base`` dictionary.

``axes``: `dict` or `list` of `dict`
    The axes of the parameter search.  The keys are "dotted object
    path" specifying the (possibly nested) parameter to be varied.
    For example, the key ``a.b.c`` means to vary the parameter
    ``param["a"]["b"]["c"]``.  The values are the list of possible
    parameters or the Python code that is evaluated to the iterable of
    possible parameters.  By default, the specified program is run for
    all the combinations of the parameters.  It can also be a `list`
    of `dict` of disjoint keys, to specify the order in which the axes
    are varied.

``format``: `str`
    The path to the file generated for ``i``-th parameter (0-origin).
    It must contain ``{i}`` which is replaced with the parameter index
    ``i``.  More precisely, this is the Python format string.


Run script file (`run.py`)
--------------------------

It must define a function `run` and/or `run_array` with the call
signature specified below.

`run(*, filepath, dirpath, filename, param, source, **_)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Function `run` must accept *any* keyword arguments, including the ones
listed here, and return a dictionary with the keys specified in the
section below.

`filepath`: `str`
    The file path to the generated parameter file; i.e., the path
    specified by ``format`` of the source parameter file.

`dirpath`: `str`
    The directory part (dirname) of `filepath`.

`filename`: `str`
    The name part (basename) of `filepath`.

`param`: `dict`
    The generated parameter.

`source`: `dict`
    The parsed source parameter file.


`run_array(*, ids, array, source, **_)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Function `run_array` is like `run` but it is run only once.  The
command that is run via `run_array` has to sweep over the set of the
parameter specified by `ids` or `array`.

`ids`: `list` of `int`
    List of the parameter index (the ``i`` in ``format``).

`array`: `str`
    The range format representation of ``ids`` that can be used for
    sbatch and qsub.  For example, it is ``"0-3"`` when `ids` is
    ``[0, 1, 2, 3]``.

`source`: `dict`
    The parsed source parameter file.


Returned `dict` from `run` and `run_array`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``"command"``: `str` or `list` of `str`
    This entry is required.  It specifies the command to run for each
    parameter.

``"input"``: `str`
    This is fed to the stdin of the `command`.

Other entries in the dictionary is passed to `subprocess.Popen`.
Following keys have specific default value.

``"shell"``: `bool`
    By default, this is `True`/`False` if `command` is a `str`/`list`.

``"cwd"``:
    For `run` function, this defaults to `dirpath` (the directory in
    which the parameter file is generated).

Keys ``"universal_newlines"`` and ``"stdin"`` cannot be set.
