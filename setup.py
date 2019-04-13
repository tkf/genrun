from setuptools import setup

import genrun

setup(
    name="genrun",
    version=genrun.__version__,
    py_modules=["genrun"],
    author=genrun.__author__,
    author_email="aka.tkf@gmail.com",
    # url='https://github.com/tkf/genrun',
    license=genrun.__license__,
    # description='genrun - THIS DOES WHAT',
    long_description=genrun.__doc__,
    # keywords='KEYWORD, KEYWORD, KEYWORD',
    classifiers=[
        "Development Status :: 3 - Alpha",
        # see: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    install_requires=["numpy", "PyYAML"],
)
