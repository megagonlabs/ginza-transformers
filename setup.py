from setuptools import setup, find_packages


setup(
    author="Megagon Labs, Tokyo.",
    author_email="ginza@megagon.ai",
    description="ginza-transformers",
    entry_points={
        "spacy_architectures": [
            "ginza-transformers.TransformerModel.v1 = ginza_transformers:architectures.TransformerModelCustom",
        ],
    },
    install_requires=[
        "spacy-transformers>=1.0.2",
    ],
    license="MIT",
    name="ginza-transformers",
    packages=find_packages(include=["ginza_transformers"]),
    url="https://github.com/megagonlabs/ginza-transformers",
    version='0.1.0',
)