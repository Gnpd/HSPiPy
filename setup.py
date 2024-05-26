from setuptools import setup


with open("README.rst", "r") as description:
    long_description = description.read()

setup(
    name="HSPiPy",
    version="0.3",
    license="MIT",
    description="Hansen Solubility Parameters in Python",
    long_description_content_type="text/x-rst",
    long_description=long_description,
    author="Alejandro Gutierrez",
    author_email="agutierrez@g-npd.com",
    url="https://github.com/Gnpd/HSPiPy",
    download_url="https://github.com/Gnpd/HSPiPy/archive/refs/tags/v_0.3.tar.gz",
    install_requires=["hspcore", "pandas", "matplotlib", "numpy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
)
