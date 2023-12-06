import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

setuptools.setup(
    name="HSPy",
    version="0.0.1",
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    description="Hansen Solubility Parameters in python",
    url="https://github.com/Gnpd/HSPy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)