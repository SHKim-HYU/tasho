import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tasho", # Replace with your own username
    version="0.0.1",
    author="Alejandro Astudillo, Ajay Sathya",
    author_email="alejandro.astudillovigoya@kuleuven.be",
    description="A small example package",
    keywords="task specification optimal control robotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.mech.kuleuven.be/meco-software/tasho",
    packages=setuptools.find_packages(),
    install_requires=[
        'casadi>=3.4,<4.0',
        'rockit-meco',
        'numpy',
        'matplotlib',
        'scipy'
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
)
