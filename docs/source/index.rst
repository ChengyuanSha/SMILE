.. LGP documentation master file, created by
   sphinx-quickstart on Mon Mar 16 16:31:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Linear Genetic Programming's documentation
==========================================

Introduction
------------

Linear Genetic Programming (LGP) is a paradigm of genetic programming that employs a representation of
linearly sequenced instructions. A population of diverse candidate models is initialized randomly and
will improve prediction accuracy gradually using random sampled training set through a number of generations.
After evolution, the best model with highest fitness score (i.e. accuracy on random sampled training set) will
be the output.

linear genetic programming package implements LGP algorithm in python, with a scikit-learn compatible API.
It retains the familiar scikit-learn `fit/predict` API and works with the existing scikit-learn modules (e.g.
`grid search <http://scikit-learn.org/stable/modules/grid_search.html>`_ ).


LGP API
-------
.. automodule:: linear_genetic_programming.lgp_classifier
   :inherited-members: score
   :members:



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



