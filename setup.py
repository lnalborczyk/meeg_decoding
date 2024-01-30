from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here/"README.md").read_text(encoding="utf-8")

setup(
    name="meg_decoding_tools",
    version="0.0.1",
    description="Open source tools for MEG preprocessing, basic analyses, and multivariate pattern analyses (aka decoding), based on MNE-Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lnalborczyk/meg_decoding_tools",
    author="Ladislas Nalborczyk",
    author_email="ladislas.nalborczyk@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3"
    ],
    # keywords="eeg, meg, mne, decoding",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    # nstall_requires=["mne"],
    project_urls={
        "Bug Reports": "https://github.com/lnalborczyk/meg_decoding_tools/issues",
        "Source": "https://github.com/lnalborczyk/meg_decoding_tools/"
    },
)