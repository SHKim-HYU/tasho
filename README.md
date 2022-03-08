# Tasho - A model predictive control toolchain for constraint-based task specification of robot motions

<div align="center">

[![pipeline status](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/pipeline.svg)](https://gitlab.mech.kuleuven.be/meco-software/tasho/commits/master)
[![coverage report](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/coverage.svg)](https://meco-software.pages.mech.kuleuven.be/tasho/coverage/index.html)
[![license: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-success.svg)](https://opensource.org/licenses/LGPL-3.0)
[![html docs](https://img.shields.io/static/v1.svg?label=docs&message=online&color=informational)](http://meco-software.pages.mech.kuleuven.be/tasho)

</div>

## Description

**Tasho** stands for "TAsk Specification with receding HOrizon control". It is a user-friendly Python toolbox that provides a unified workflow that streamlines and automates many steps from specification to experimental deployment to enable rapid prototyping of OCP/MPC based motion skills in robotics.

(Note: Documentation and examples are under heavy development. Will be stable by March 12th, 2022. Thank you for your patience.)

**Authors:** [Ajay Sathya](https://scholar.google.com/citations?hl=es&user=A00LDswAAAAJ) and [Alejandro Astudillo](https://scholar.google.com/citations?user=9ONkJZAAAAAJ).  
**With support from [Joris Gillis](https://scholar.google.com/citations?hl=es&user=sQtYwmgAAAAJ), [Wilm Decré](https://scholar.google.com/citations?hl=es&user=ZgAnArUAAAAJ), [Goele Pipeleers](https://scholar.google.com/citations?hl=es&user=TKWS1vEAAAAJ) and [Jan Swevers](https://scholar.google.com/citations?hl=es&user=X_fnO1YAAAAJ) from the [MECO Research Team](https://www.mech.kuleuven.be/en/pma/research/meco/) at KU Leuven, Belgium.**

**License:** Tasho is released under the [GNU LGPLv3 license](LICENSE).

## Installation

### Option 1: Installing with pip
You can install Tasho (ideally into a virtual environment) via pip using the following command:

```
pip install git+https://gitlab.kuleuven.be/meco-software/tasho.git@main
```

### Option 2: Installing from cloned repository
Alternatively, you can clone this repository and install Tasho from source. You just need to (i) clone the repository, (ii) move into Tasho's root directory, and (iii) run the `setup.py` script with the `install` option. It will install your application into the virtualenv site-packages folder and also download and install all dependencies:

```
git clone https://gitlab.kuleuven.be/meco-software/tasho.git
cd tasho
python setup.py install
```
You could also use the `develop` option, instead of `install`, during the execution of `setup.py` as `python setup.py develop`. 
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then modify/update the source code without having to run `python setup.py install` again after every change.

## Examples

Several examples are provided with Tasho. The following examples list is arranged in ascending order of complexity. (Uncomment the GUI option for visualization.)
- Hello world: [examples/hello_world_AT.py](examples/hello_world_AT.py)
- Point-to-point motion: [examples/P2P_AT.py](examples/P2P_AT.py)
- Tunnel-following optimal control: [examples/pose_tunnel_AT.py](examples/pose_tunnel_AT.py)
- Bouncing ball: [examples/bouncing_ball.py](examples/bouncing_ball.py)

## Submitting an issue

Please submit an issue if you want to report a bug or propose new features.
