#!/usr/bin/env python3
"""
Setup script for Scraping CLI package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A command-line interface for web scraping using CrewAI and Browserbase."

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="scraping-cli",
    version="0.1.0",
    author="Scraping CLI Team",
    author_email="team@scraping-cli.com",
    description="A command-line interface for web scraping using CrewAI and Browserbase",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/scraping-cli/scraping-cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "scraping-cli=scraping_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="scraping, cli, crewai, browserbase, web automation",
    project_urls={
        "Bug Reports": "https://github.com/scraping-cli/scraping-cli/issues",
        "Source": "https://github.com/scraping-cli/scraping-cli",
        "Documentation": "https://github.com/scraping-cli/scraping-cli#readme",
    },
) 