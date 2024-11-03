from pathlib import Path

from setuptools import setup, find_packages

with open(Path(__file__).parent / "VERSION", encoding="utf-8") as f:
    version = f.read()

with open(Path(__file__).parent / "README", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gym-dmc",
    packages=find_packages(exclude=["specs", "notebooks"]),
    # gym 0.21.0 requires the lower pip version. Modifying pip version is really not ideal.
    install_requires=[
        "dm_control",
        "numpy",
    ],
    # update: removed gym
    #
    # gym requires
    #
    # > pip install setuptools==65.5.0
    # > pip install wheel==0.38.4
    # > pip install gym-dmc
    description="gym-dmc is a gym wrapper around DeepMind Control Suite domains.",
    long_description=long_description,
    author="Ge Yang<ge.ike.yang@gmail.com>",
    url="https://github.com/geyang/gym-dmc",
    author_email="ge.ike.yang@gmail.com",
    version=version,
)
