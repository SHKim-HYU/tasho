# TASHO - A model predictive control toolchain for constraint-based task specification of robot motions
[![pipeline status](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/pipeline.svg)](https://gitlab.mech.kuleuven.be/meco-software/tasho/commits/master)
[![coverage report](https://gitlab.mech.kuleuven.be/meco-software/tasho/badges/master/coverage.svg)](https://gitlab.mech.kuleuven.be/meco-software/tasho/coverage/index.html)



### Installing/Developing
To install your application (ideally into a virtualenv) just run the setup.py script with the install parameter. It will install your application into the virtualenv site-packages folder and also download and install all dependencies:
```
python setup.py install
```

If you are developing on the package and also want the requirements to be installed, you can use the develop command instead:
```
python setup.py develop
```
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then continue to work on the code without having to run install again after each change.
