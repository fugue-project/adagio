from setuptools import setup, find_packages

VERSION = "0.1.0"

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="adagio",
    version=VERSION,
    packages=find_packages(),
    description="A Dag IO Framework",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="adagio dag directed acyclic graph",
    url="http://github.com/goodwanghan/adagio",
    install_requires=["triad>=0.1.3"],
    extras_require={},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3",
)
