from setuptools import setup

setup(
    name='sklearn_x_rdkit',
    version='0.0.1.1',
    author='Christian Feldmann',
    license="BSD",
    packages=['sklearn_x_rdkit', 'tests'],
    author_email='cfeldmann@bit.uni-bonn.de',
    description='Convenient transfer from rdkit features to sklearn input',
    install_requires=['scipy', 'bidict', 'numpy']
)
