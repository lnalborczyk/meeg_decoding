from setuptools import setup, find_packages

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"data": ["*.fif"]}
)
