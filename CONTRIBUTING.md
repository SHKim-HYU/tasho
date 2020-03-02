# Contributing
If you want to contribute to the project and make it better, your help is very welcome.

### Submitting an issue
Please submit an issue if you want to report a bug or propose new features.

### Installing/Developing
To install the toolchain (ideally into a virtualenv) just run the setup.py script with the install parameter. It will install your application into the virtualenv site-packages folder and also download and install all dependencies:
```
python setup.py install
```

If you are developing on the toolchain and also want the requirements to be installed, you can use the develop command instead:
```
python setup.py develop
```
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then continue to work on the code without having to run install again after each change.
