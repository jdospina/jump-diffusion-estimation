from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jump-diffusion-estimation",
    license="MIT",
    version="0.1.0",
    author="Juan David OSPINA ARANGO",
    author_email="jdospina@gmail.com",
    description="A dream about (someday) a comprehensive library for jump-diffusion parameter estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdospina/jump-diffusion-estimation",
    project_urls={
        "Bug Tracker": "https://github.com/jdospina/jump-diffusion-estimation/issues",
        "Documentation": "https://jump-diffusion-estimation.readthedocs.io/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "sphinx",
            "sphinx-rtd-theme",
            "pandas-stubs",
            "scipy-stubs",
        ],
        "tutorials": [
            "jupyter",
            "matplotlib",
            "seaborn",
            "plotly",
        ],
    },
    entry_points={
        "console_scripts": [
            "jumpdiff-validate=jump_diffusion.scripts.validate:main",
            "jumpdiff-benchmark=jump_diffusion.scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
