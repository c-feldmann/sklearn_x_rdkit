from setuptools import setup

setup(
    name='sklearn_x_rdkit',
    version='0.0.1',
    packages=[''],
    url='',
    license='',
    author='Christian Feldmann',
    author_email='cfeldmann@bit.uni-bonn.de',
    description='Convenient transfer from rdkit features to sklearn input', install_requires=['rdkit', 'scipy',
                                                                                              'bidict']
)
