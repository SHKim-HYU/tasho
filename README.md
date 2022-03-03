# TASHO - A model predictive control toolchain for constraint-based task specification of robot motions

<div align="center">

[![pipeline status](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/pipeline.svg)](https://gitlab.mech.kuleuven.be/meco-software/tasho/commits/master)
[![coverage report](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/coverage.svg)](https://meco-software.pages.mech.kuleuven.be/tasho/coverage/index.html)
[![license: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-success.svg)](https://opensource.org/licenses/LGPL-3.0)
[![html docs](https://img.shields.io/static/v1.svg?label=docs&message=online&color=informational)](http://meco-software.pages.mech.kuleuven.be/tasho)

</div>

## Description

**Tasho** stands for "TAsk Specification with receding HOrizon control".

### Installing

**Option 1**: You can install Tasho (ideally into a virtual environment) via pip using the following command:

```
pip install git+https://gitlab.kuleuven.be/meco-software/tasho.git@main
```

**Option 2**: Alternatively, you can clone this repository and install Tasho from source. You just need to (i) clone the repository, (ii) move into Tasho's root directory, and (iii) run the `setup.py` script with the `install` option. It will install your application into the virtualenv site-packages folder and also download and install all dependencies:

```
git clone https://gitlab.kuleuven.be/meco-software/tasho.git
cd tasho
python setup.py install
```
You could also use the `develop` option, instead of `install`, during the execution of `setup.py` as `python setup.py develop`. 
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then modify/update the source code without having to run `python setup.py install` again after every change.

### Submitting an issue

Please submit an issue if you want to report a bug or propose new features.
