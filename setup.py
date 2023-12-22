from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="HSPiPy",
    version="0.0.2",
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    description="Hansen Solubility Parameters in Python",
    long_description=long_description,
    long_description_content_type='text/markdown'
    url="https://github.com/Gnpd/HSPiPy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy'],
    python_requires='>=3.6',
)