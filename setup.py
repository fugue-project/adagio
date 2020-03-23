from setuptools import setup, find_packages

VERSION = "0.0.2"

setup(
    name="adagio",
    version=VERSION,
    packages=find_packages(),
    description="A Dag IO Framework",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    install_requires=["pandas"],
    extras_require={},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        # Define that your audience are developers
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3",
)
