from setuptools import setup, find_packages

setup(
    name="zmb-classifiers",
    version="0.1",
    packages=find_packages(include=["zmb_classifiers", "zmb_classifiers.*"]),
    install_requires=[
        "pandas",
        "scikit-learn",
        "PyYAML",
        "datasets",
        "transformers",
        "torch",
        "huggingface_hub",
        "safetensors"
    ],
    entry_points={
        'console_scripts': [
            'zmb-clf = zmb_classifiers.cli:main',
        ],
    },
    author="Jefferson O. Silva",
    description="Classificador que determina se uma matéria de jornal contém referências às raças negra, branca ou a elementos da cultura negra.",
    python_requires=">=3.8",
)
