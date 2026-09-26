from setuptools import setup, find_packages

setup(
    name="sutte-arima",
    version="1.0.0",
    author="Bahman Orujov",
    author_email="orucovbehmen@gmail.com",
    description="SutteARIMA: Statsmodels-style hybrid Alpha-Sutte + ARIMA time series forecasting model",
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
)
