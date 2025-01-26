from setuptools import setup, find_packages, find_namespace_packages

setup(
    name="HiStrux",
    version="1.0.0",
    author="Adrian Zaręba, Wiktor Wierzchowski",
    author_email="zarebaff50@gmail.com, w.wierzch@gmail.com",
    description="Python package implementing tools for working with Hi-C features extraction and reconstructions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AdixPlaysGames/HiStrux",
    install_requires=[
        'pandastable>=0.13.1 '
    ],

    # packages=find_namespace_packages(where="package"),
    # package_dir={"": "package"},

    packages=find_packages(include=['HiStrux', 'HiStrux.eXtract', 'HiStrux.CycleSort', 'HiStrux.reConstruct']),

    license="MIT",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)