import setuptools

NAME = "tasho"
VERSION = "0.0.1"
AUTHORS = "Alejandro Astudillo, Ajay Sathya"
MAINTAINER_EMAIL = "alejandro.astudillovigoya@kuleuven.be"
DESCRIPTION = "TASHO - A model predictive control toolchain for constraint-based task specification of robot motions"
KEYWORDS = "task specification optimal control robotics"
URL = "https://gitlab.kuleuven.be/meco-software/tasho"
LICENSE = 'GNU LGPLv3'

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    license=LICENSE,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=[
        'rockit-meco>=0.1.28',
        'casadi>=3.4,<4.0',
        'numpy',
        'matplotlib',
        'scipy',
        'pybullet',
        'robotsMECO @ git+https://gitlab.kuleuven.be/meco-software/robot-models-meco.git@main'
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
