HSPiPy
======

Hansen Solubility Parameters in Python.
---------------------------------------

Introduction
----------------

HSPiPy is a Python library designed for calculating and visualizing
Hansen Solubility Parameters (HSP). The library provides tools to
compute HSP from a grid of solvent data and offers 2D and 3D plotting
capabilities to visualize the solubility parameter space

Installation
^^^^^^^^^^^^^^^

Install **HSPiPy** easily with pip:

::

   pip install HSPiPy

Usage
^^^^^^^^^^^^^^^

Reading HSP Data
----------------

To read HSP data from a CSV file, create an instance of the ``HSP``
class and use the ``read`` method:

.. code:: python

   from hspipy import HSP

   hsp = HSP()
   hsp.read('path_to_your_hsp_file.csv')

Calculating HSP
----------------

Use the ``get`` method to calculate the Hansen Solubility Parameters
(HSP) from your data:

.. code:: python

   hsp.get()

Visualizing HSP
----------------

Use the ``plot_3d`` and ``plot_2d`` methods
to visualize the HSP data in 3D and 2D formats, respectively:

.. code:: python

   hsp.plot_3d()
   hsp.plot_2d()

``HSP`` class methods:
----------------------

+---------------------+---------------------------------------------------------+
| Method              | Description                                             |
+=====================+=========================================================+
| read(path)          | Reads solvent data from a CSV file.                     |
+---------------------+---------------------------------------------------------+
| get(inside_limit=1) | Calculates the HSP and identifies solvents inside and   |
|                     | outside the solubility sphere.                          |
+---------------------+---------------------------------------------------------+
| plot_3d()           | Plots the HSP data in 3D.                               |
+---------------------+---------------------------------------------------------+
| plot_2d()           | Plots the HSP data in 2D.                               |
+---------------------+---------------------------------------------------------+
| plots()             | Generates both 2D and 3D plots.                         |
+---------------------+---------------------------------------------------------+

Once you have calculated the HSP parameters using the get() method, you
can access the calculated HSP parameters and related attributes through
the properties of the HSP class instance. Below are the attributes you
can access:

+----------------+---------------------------------------------------------+
| Attribute      | Description                                             |
+================+=========================================================+
| ``hsp.d``      | Float - Dispersion parameter of the HSP.                |
+----------------+---------------------------------------------------------+
| ``hsp.p``      | Float - Polar parameter of the HSP.                     |
+----------------+---------------------------------------------------------+
| ``hsp.h``      | Float - Hydrogen-bonding parameter of the HSP.          |
+----------------+---------------------------------------------------------+
| ``hsp.radius`` | Float - Radius of the solubility sphere.                |
+----------------+---------------------------------------------------------+
| ``hsp.error``  | Float - Error in the HSP calculation.                   |
+----------------+---------------------------------------------------------+
| ``hsp.inside`` | Numpy array - Solvents inside the solubility sphere.    |
+----------------+---------------------------------------------------------+
| ``hsp.outside``| Numpy array - Solvents outside the solubility sphere.   |
+----------------+---------------------------------------------------------+
| ``hsp.grid``   | A Pandas DataFrame containing the solvent data with     |
|                | columns for the solvent name, dispersion (D), polar (P),|
|                | hydrogen-bonding (H), and score values.                 |
+----------------+---------------------------------------------------------+

Contributing
^^^^^^^^^^^^^^^

Contributions are welcome! If you have any suggestions, feature
requests, or bug reports, please open an issue on the `GitHub
repository <https://github.com/Gnpd/HSPiPy/issues>`__.

License
^^^^^^^^^^^^^^^

This library is licensed under the MIT License. See the
`LICENSE <https://github.com/Gnpd/HSPiPy/blob/main/LICENSE>`__ file for
details.

Acknowledgements
^^^^^^^^^^^^^^^^^

HSPiPy was inspired by the well-known HSP software suit `Hansen
Solubility Parameters in Practice
(HSPiP) <https://www.hansen-solubility.com/HSPiP/>`__ and by the HSP
community.

