from os import path
from setuptools import setup, find_packages

with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(name='gym-dmc',
      packages=find_packages(),
      install_requires=[
          "gym",
          "numpy",
      ],
      description='gym-dmc',
      author='Ge Yang<ge.ike.yang@gmail.com>',
      url='https://github.com/geyang/gym_dmc',
      author_email='ge.ike.yang@gmail.com',
      version=version)
