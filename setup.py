from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="HSPiPy",
    version="0.1",
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    description="Hansen Solubility Parameters in Python",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Gnpd/HSPiPy",
    download_url="https://github.com/Gnpd/HSPiPy/archive/refs/tags/0.1.tar.gz",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=['hspcore','pandas','matplotlib','numpy'],
    python_requires='>=3.6',
)