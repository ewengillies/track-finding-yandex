"""
Install comet-ml
"""
from setuptools import setup

# Define the minimal classes needed to install and run tigramite
INSTALL_REQUIRES = [
    "numpy",
    "uproot",
    "pandas",
    "dask",
    "scipy",
    "sklearn",
    "matplotlib",
    "pprint",
    "tabulate",
    "pytest"]

# Run the setup
setup(
    name='comet-ml',
    version='0.0.0-beta',
    packages=['cometml'],
    license='GNU General Public License v3.0',
    description='ML based track reconstruction for the COMET experiment',
    author='Ewen Gillies',
    author_email='e.gillies.ix@gmail.com',
    install_requires=INSTALL_REQUIRES,
    test_suite='tests'
)
