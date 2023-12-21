import os
from read_version import read_version
from setuptools import setup, find_packages
from pathlib import Path

here = os.path.abspath(os.path.dirname(__file__))

os.chdir(here)

version = read_version('eyepop', '__init__.py')
long_description = (Path(here) / "README.md").read_text()

setup(
    name="eyepop-sdk-python",
    version=version,
    description="EyePop.ai Python SDK",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="EyePop.ai",
    author_email="info@eyepop.ai",
    url="https://github.com/eyepop-ai/eyepop-sdk-python",
    license="MIT",
    keywords="EyePop AI ML CV",
    packages=find_packages(include=["eyepop"], exclude=["tests", "tests.*"]),
    install_requires=[
        'aiohttp >= 3.9.1',
        'matplotlib >= 3.8.2'
    ],
    python_requires=">=3.8",
    project_urls={
        "Bug Tracker": "https://github.com/eyepop-ai/eyepop-sdk-python/issues",
        "Changes": "https://github.com/eyepop-ai/eyepop-sdk-python/blob/master/CHANGELOG.md",
        "Documentation": "https://github.com/eyepop-ai/eyepop-sdk-python",
        "Source Code": "https://github.com/eyepop-ai/eyepop-sdk-python",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    setup_requires=["wheel"],
)