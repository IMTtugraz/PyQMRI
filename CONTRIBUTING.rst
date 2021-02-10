How to contribute to PyQMRI
===========================
Thank you for taking your time to consider to contribute to PyQMRI, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features

If you want to contribute please first contact the owner of this repository (e.g. via opening an Issue) and discuss the changes you want to change.
By contributing you accept and adhere to our `Code of Conduct`_
Development and code contributaions should be done at our GitLab_ site to facilitate the CI integration and GPU availability there.
Please contact us for further information.

Found a Bug?
------------

* **Search if someone else already reported that bug**
* **If not, open a new GitHub Issue_** and describe the steps that lead to the issue as detailed as possible. Use a clear title and add as much information as possible to reproduce the issue.

We Use `Github Flow`_, So All Code Changes Happen Through Pull Requests
-----------------------------------------------------------------------

Pull requests are the best way to propose changes to the codebase (we use `Github Flow`_). We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

By contributing you accept that any contributions you make will be under the `Apache-2`_ license. Feel free to contact the maintainers if that's a concern to you. 

.. role:: bash(code)
   :language: bash
.. role:: python(code)
   :language: python
   
Contributing Code/Fixing bugs
-----------------------------
For code contributions, it is mandatory that you make sure that all current unittests and integrationtest pass after your changes. 

Prior to running the tests it is necessary to start an ipcluster. 
An exemplary workflow would be:
:bash:`ipcluster start &`
followed by typing
:bash:`pytest test`
in the PyQMRI root folder. It is advised to run unit and integration tests after each other as OUT_OF_MEMORY exceptions can occur if both are in one session, e.g.:
:bash:`pytest test/unittests`
:bash:`pytest test/integrationtests`

If you add new functionality it is also mandatory to provide unit/integrationtests to make sure that the new code provides the expected behaviour.
Also make sure that the code you submit adherse to the PEP8_ coding style for python and all public functions are documented.

References
----------
This document was adapted from the open-source contribution guidelines for `Facebook's Draft`_.

.. _`Github Flow` : https://guides.github.com/introduction/flow/index.html
.. _Issue : https://github.com/IMTtugraz/PyQMRI/issues
.. _PEP8 : https://www.python.org/dev/peps/pep-0008/
.. _`Facebook's Draft` : https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md
.. _`Apache-2` : LICENSE
.. _`Code of Conduct` : CODE_OF_CONDUCT.rst
