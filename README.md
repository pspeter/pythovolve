![travis build](https://img.shields.io/travis/peter-schmidbauer/pythovolve.svg)
![PyPI - License](https://img.shields.io/pypi/l/pythovolve.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pythovolve.svg)

# pythovolve
A modular, object-oriented framework for evolutionary and genetic algorithms in Python 3.

## Quick start

### Installation using pip

Make sure you have Python 3.6 installed by executing `python --version` in your command line.

Next, simply execute 

    pip install --upgrade pythovolve 
    
to install pythovolve.

### Try it out (as a Library)

Check out the examples in the examples directory. To do that, clone the repository using git:

    git clone https://github.com/peter-schmidbauer/pythovolve.git

If you have already installed pythovolve, you can now run

    python pythovolve/examples/<example_script.py>
    
to execute one of the examples.

### Try it out (as a CLI)

If you have already installed pythovolve, check out a simple CLI example by running:

    python -m pythovolve GA -r 30 -p
    
To run an ES on a difficult multi dimensional test function, try

    python -m pythovolve ES -d hoelder_table -m gauss -c single_point -p
    
For a full list and explanation of all CLI parameters, run

    python -m pythovolve -h
    


    
