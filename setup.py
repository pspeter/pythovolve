from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(name='pythovolve',
      version='0.1.1',
      description='Object oriented framework for genetic algorithms',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/peter-schmidbauer/pythovolve',
      author='Peter Schmidbauer',
      author_email='peter.schmidb@gmail.com',
      license='MIT',
      packages=['pythovolve'],
      install_requires=["matplotlib", "seaborn", "sympy", "pandas", "scipy"],
      zip_safe=False,
      classifiers=(
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
      ))
