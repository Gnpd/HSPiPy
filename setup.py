import os
import setuptools

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

setuptools.setup(
    name="HSPiPy",
    version="0.0.1",
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    description="Hansen Solubility Parameters in Python",
    url="https://github.com/Gnpd/HSPy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy'],
    python_requires='>=3.6',
)