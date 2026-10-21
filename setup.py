from setuptools import setup, find_packages

setup(
    name="sutte-arima",
    version="2.0.0",
    author="Bahman Orujov",
    author_email="orucovbehmen@gmail.com",
    description="SutteARIMA: statsmodels-style hybrid alpha-Sutte + ARIMA forecasting with leakage-free evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BahmanOrujov/sutte-ARIMA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.2",
        "scipy>=1.6",
        "statsmodels>=0.12",
    ],
    extras_require={"plot": ["matplotlib"], "test": ["pytest"]},
)
