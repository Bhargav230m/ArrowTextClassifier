from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ArrowTextClassifier",
    version="1.0.1",
    author="techpowerb",
    author_email="technologypower24@gmail.com",
    description="ArrowTextClassifier is a simple text classification tool written in pytorch that allows you to train, summarize, and use text classification models for various tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bhargav230m/ArrowTextClassifier.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch==2.2.2",
        "torchsummary==1.5.1",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.2",
        "numpy==1.26.4",
    ],
    keywords=[
        "text classification",
        "natural language processing",
        "NLP",
        "PyTorch",
        "machine learning",
        "deep learning",
        "text summarization",
        "preprocessing",
        "data science",
        "artificial intelligence",
        "dataset",
        "discord",
    ],
)
