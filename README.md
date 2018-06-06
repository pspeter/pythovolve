# pythovolve
A modular, object-oriented framework for evolutionary and genetic algorithms in Python 3.

## Quick start

### Installation using pip

Make sure you have Python 3.6 installed by executing `python --version` in your command line.

Next, simply execute 

    pip install --upgrade pythovolve 
    
to install pythovolve.

### Try it out (as a Library)

Check out the examples in the examples directory. 
Simply run any of them as a script with pythovolve installed.

### Try it out (as a CLI)

To check out a simple CLI example, run 

    python -m pythovolve GA -r 30 -p
    
To run an ES on a difficult multi dimensional test function, try

    python -m pythovolve ES -d hoelder_table -m gauss -c single_point -p
    
For a full list and explanation of all CLI parameters, run

    python -m pythovolve -h
    


    