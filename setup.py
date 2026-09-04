from setuptools import setup, find_packages


setup(
    author="Megagon Labs, Tokyo.",
    author_email="ginza@megagon.ai",
    description="ginza-transformers",
    entry_points={
        "spacy_factories": [
            "transformer_custom = ginza_transformers.pipeline_component:make_transformer_custom",
        ],
    },
    install_requires=[
        "ginza>=5.3.0",
        "spacy-transformers==1.4.0",
    ],
    extras_require={
        "cpu": [],
        "apple": ["thinc-apple-ops"],
        "cuda11x": ["cupy-cuda11x"],
        "cuda12x": ["cupy-cuda12x"],
    },
    license="MIT",
    name="ginza-transformers",
    packages=find_packages(include=["ginza_transformers", "ginza_transformers.layers"]),
    url="https://github.com/megagonlabs/ginza-transformers",
    version='1.4.0',
)
