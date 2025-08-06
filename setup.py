from setuptools import setup, find_packages


with open("README.rst", "r") as description:
    long_description = description.read()

setup(
    name="HSPiPy",
    version="0.3.3",
    license="MIT",
    description="Hansen Solubility Parameters in Python",
    long_description_content_type="text/x-rst",
    long_description=long_description,
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    url="https://github.com/Gnpd/HSPiPy",
    download_url="https://github.com/Gnpd/HSPiPy/archive/refs/tags/v_0.3.1.tar.gz",
    packages=find_packages(),
    package_data={
        'hspipy': ['*.csv', '*.png'],
    },
    install_requires=[
        "scipy>=1.0.0",  # Fixed from scipy.optimize
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.18.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    keywords="hansen solubility parameters chemistry solvents",
)

