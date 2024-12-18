# GRFski is python script to compute the ski triggering probability on virtual slope
This script is based on the publications of [Meloche et al. (2024)][Meloche2024] which compute the skier triggering probability on spatially variable virtual slope.
The script can also count the number of skiers until trigger based on two approaches (random vs structured). This was based on the idea that skiing near a prexisting skier track is safer.

The slab depth is spatially variable using a Gaussian Random Field GRF using 3 parameters:
1. Mean slab depth
2. Variance slab depth
3. Correlation length slab depth

The slab density and elastic modulus are set to the mean slab depth vi empirical relationship
1. Slab density [McClung (2009)][McClung2009]
2. Elastic modulus [Sigrist (2006)][Sigrist2006]
    
The weak layer shear strength is set to the local slab depth using an derived empirical/Mohr-Colomb relation
1. Weak layer shear strength [Gaume and Reuter (2017)][Gaume2017]
    
The virtual ski slope is set to 35 degree
The skier loading (R) is set to 780 N from Gaume and Reuter 2017

# Technical details
## Requirements
This package requires:

* Python, with version 3.8 at least
* Additional packages:
    * `numba`
    * `numpy`
    * `gstools`
    * `matplotlib` if you plan to run the examples
    * `pandas` to store et extract the results

  ## Usage
  The function are under the python script GRFski.py
  Examples are provided with the Ipython notebook with two types of examples
  1. Compute the skier probability
  2. Count the number of "safe skier" until trigger with a structed and a random approach
  
# Authorship and Licence

This code is the result of the work of :

* Francis Meloche
* Louis Guillet
* Johan Gaume

We thank the users to credit the original authors of this code, for instance by citing [Meloche et al. (2024)][Meloche2024].

This piece of code is provided with absolutely no warranty.

[Meloche2024]: https://doi.org/10.1017/aog.2024.3
[McClung2009]: https://doi.org/10.1029/2007JF000941
[Sigrist2006]: https://doi.org/10.3929/ethz-a-005282374
[Gaume2017]: https://doi.org/10.1016/j.coldregions.2017.05.011
