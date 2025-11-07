"""
TELOS - Runtime AI Governance System
Setup configuration for Python package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="telos",
    version="1.0.0",
    author="TELOS Labs",
    author_email="telos.steward@gmail.com",
    description="Runtime AI Governance System with Dual Primacy Attractor Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TelosSteward/TELOS",
    packages=find_packages(include=["telos", "telos.*", "observatory", "observatory.*", "steward", "steward.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "telos-observatory=observatory.main:main",
            "telos-steward=steward.steward:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
