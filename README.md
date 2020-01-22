# TASHO - A model predictive control toolchain for constraint-based task specification of robot motions

### Installing/Developing
To install your application (ideally into a virtualenv) just run the setup.py script with the install parameter. It will install your application into the virtualenv’s site-packages folder and also download and install all dependencies:
```
python setup.py install
```

If you are developing on the package and also want the requirements to be installed, you can use the develop command instead:
```
python setup.py develop
```
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then continue to work on the code without having to run install again after each change.
