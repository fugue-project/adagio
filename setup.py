from adagio_version import __version__
from setuptools import find_packages, setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="adagio",
    version=__version__,
    packages=find_packages(),
    description="The Dag IO Framework for Fugue projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="adagio dag directed acyclic graph workflow",
    url="http://github.com/fugue-project/adagio",
    install_requires=["triad>=0.4.3"],
    extras_require={},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
)
