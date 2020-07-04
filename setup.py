import setuptools

long_description = "Check github (https://github.com/ChengyuanSha/linear_genetic_programming) for details"

setuptools.setup(
    name="lgp",
    version="0.0.2",
    author="Chengyuan Sha",
    author_email="15cs69@queensu.ca",
    description="A python implementation of linear genetic programming algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChengyuanSha/linear_genetic_programming",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.1',
        'pandas>=0.25.3',
        'scikit-learn>=0.22.1',
        'scipy>=1.3.1'
    ],
)
