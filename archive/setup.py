"""
TELOS Setup Configuration
==========================

Installation script for the TELOS package.

Usage:
    pip install -e .          # Development install
    pip install .             # Standard install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="telos-purpose",
    version="1.0.0",
    author="Origin Industries PBC / TELOS Labs LLC",
    author_email="research@origin-industries.org",
    description="Mathematical runtime framework for AI governance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/origin-industries/telos",
    project_urls={
        "Documentation": "https://origin-industries.org/telos",
        "Source": "https://github.com/origin-industries/telos",
        "Tracker": "https://github.com/origin-industries/telos/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "validation_results"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "matplotlib>=3.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "telos-validate=telos_purpose.validation.run_internal_test0:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ai-governance",
        "ai-alignment",
        "runtime-control",
        "llm-safety",
        "proportional-control",
        "dynamical-systems",
    ],
)
