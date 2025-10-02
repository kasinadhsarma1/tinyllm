#!/usr/bin/env python3
"""
Setup script for TinyLLM package
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "TinyLLM - A lightweight language model library with SafeTensors support"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.21.0',
        'safetensors>=0.3.0',
    ]

setup(
    name="tinyllm",
    version="1.0.0",
    author="TinyLLM Developer",
    author_email="developer@tinyllm.dev",
    description="A lightweight language model library with SafeTensors support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tinyllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tinyllm-demo=examples.package_demo:main",
            "tinyllm-chat=examples.interactive_chat:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tinyllm": ["*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/tinyllm/issues",
        "Source": "https://github.com/your-username/tinyllm",
        "Documentation": "https://tinyllm.readthedocs.io/",
    },
    keywords="llm language-model safetensors machine-learning nlp",
    zip_safe=False,
)
