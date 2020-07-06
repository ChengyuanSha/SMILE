.. image:: https://readthedocs.org/projects/linear-genetic-programming/badge/?version=latest
    :target: https://linear-genetic-programming.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://api.codacy.com/project/badge/Grade/c8897f8173434a8798896a8f94d0c2c0
    :target: https://www.codacy.com/manual/ChengyuanSha/linear_genetic_programming?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ChengyuanSha/linear_genetic_programming&amp;utm_campaign=Badge_Grade
.. image:: https://img.shields.io/website?up_message=result%20visualization&url=https%3A%2F%2Flgp-result.herokuapp.com%2F
    :target: https://lgp-result.herokuapp.com/

Welcome to Linear Genetic Programming!
======================================
**Linear genetic programming** package implements LGP algorithm in python, with a scikit-learn style API. It is
mainly used in data mining and finding feature interactions. Note it currently only support binary classification data.

|

Documentation: `here1 <http://linear-genetic-programming.rtfd.io/>`_

Result Visualization Website: `here2 <https://lgp-result.herokuapp.com/>`_

Installation
------------
This package is published on pypi. Install using pip.

.. code-block:: python

    pip install lgp

Running
-------
This algorithm is **computationally expensive**, and it needs to run approximately 1000 times parallel to produce enough
data to analyze. it needs to run in computer clusters like `compute canada. <https://www.computecanada.ca/>`_

Sample running python file (Run.py):

.. code-block:: python

    X, y, names # get X, y, names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # set own parameter here
    lgp = LGPClassifier(numberOfInput = X_train.shape[1], numberOfVariable = 200, populationSize = 20,
                            fitnessThreshold = 1.0, max_prog_ini_length = 40, min_prog_ini_length = 10,
                            maxGeneration = 2, tournamentSize = 4, showGenerationStat=False,
                            isRandomSampling=True, maxProgLength = 500)
    lgp.fit(X_train, y_train)
    y_pred = lgp.predict(X_test)
    lgp.testingAccuracy = accuracy_score(y_pred, y_test)
    lgp.save_model()

Example Bash running script:

.. code-block:: console

    #!/bin/bash
    #SBATCH --time=10:00:00
    #SBATCH --array=1-1000
    #SBATCH --mem=500M
    #SBATCH --job-name="lgp"

    python Run.py

Reference
---------
Linear_Genetic_Programming_.
Authors: Brameier, Markus F., Banzhaf, Wolfgang

.. _Linear_Genetic_Programming: https://www.springer.com/gp/book/9780387310299