from setuptools import find_packages, setup

setup(
    name="Wafer",
    version="0.0.1",
    author="anjali",
    author_email="an.anjalirai98@gmail.com",
    packages=find_packages(),
    install_requires=['Cython>=0.29.30',],
    setup_requires=['Cython>=0.29.30',  # Required for building extensions
    ],
)