from pathlib import Path

from setuptools import setup, find_packages

with open(Path(__file__).parent / "VERSION", encoding="utf-8") as f:
    version = f.read()

with open(Path(__file__).parent / "README", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gym-dmc",
    packages=find_packages(exclude="specs"),
    install_requires=[
        # gym 0.21.0 requires the lower pip version. Modifying pip version is really not ideal.
        "setuptools==65.5.1",
        "gym==0.21.0",
        "dm_control",
        "numpy",
    ],
    description="gym-dmc is a gym wrapper around DeepMind Control Suite domains.",
    long_description=long_description,
    author="Ge Yang<ge.ike.yang@gmail.com>",
    url="https://github.com/geyang/gym-dmc",
    author_email="ge.ike.yang@gmail.com",
    version=version,
)
