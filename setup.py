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
    keywords="eeg, meg, mne, decoding",
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    packages=find_packages(where="src/"),
    package_dir={"": "src"},
    # package_dir = "src",
    # packages=find_packages(),
    # packages=['jr', 'jr.tests',
    #     'jr.cloud', 'jr.cloud.tests',
    #     'jr.gat', 'jr.gat.tests',
    #     'jr.gif', 'jr.gif.tests',
    #     'jr.meg', 'jr.meg.tests',
    #     'jr.model',
    #     'jr.plot',
    #     'jr.stats', 'jr.stats.tests'],
    python_requires=">=3.9",
    install_requires=["mne"],
    project_urls={
        "Bug Reports": "https://github.com/lnalborczyk/meg_decoding_tools/issues",
        "Source": "https://github.com/lnalborczyk/meg_decoding_tools/"
    },
)