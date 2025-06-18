from setuptools import setup, find_packages

setup(
    name="zmb-classifiers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "PyYAML",
        "datasets",
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "zmb-cli=zmb.cli:main",
        ],
    },
    author="Seu Nome",
    description="Classificador de matérias jornalísticas com pipeline configurável",
    python_requires=">=3.8",
)
