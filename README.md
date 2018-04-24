# The COBAIN code

COBAIN (COntact Binary Atmospheres with INterpolation) is a generalized radiative transfer code for computation of atmosphere tables of differentially rotating and contact binary stars.

## Getting Started

To install, run
```commandline
python setup.py build
python setup.py install
```
If the installation goes well, try importing to check for potential missing prerequisites.

### Prerequisites

To run COBAIN properly, you will need to have numpy, scipy, astropy and quadpy installed on your machine. If you don't, install them via pip (or brew if on a Mac).


### Example

Creating a differentially rotating or contact binary star and populating the initial radiative transfer arrays is done with:
```python
import cobain

dr = cobain.bodies.star.DiffRot(mass=1.0, radius=1.0, teff=6000., n=3.0, 
                                dims=[10,10,10], bs=[0.1,0.,0.], pot_range=0.01, lebedev_ndir=5,
                                dir='diffrot/')
cb = cobain.bodies.binary.Contact_Binary(mass1 = 1.0, q = 0.5, ff = 0.1, pot_range = 0.01,
                                  dims = [10,10,10], n1 = 3., n2 = 3.,
                                  lebedev_ndir = 5,
                                  dir = 'contact/')
```
For more check out the example scripts.
## Authors

* **Angela Kochoska** - [gecheline](https://github.com/gecheline)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

