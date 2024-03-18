# pyLemur

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/const-ae/pyLemur/test.yaml?branch=main
[link-tests]: https://github.com/const-ae/pyLemur/actions/workflows/test.yaml
[link-docs]: https://pyLemur.readthedocs.io
[badge-docs]: http://readthedocs.org/projects/pylemur/badge

The Python implementation of the LEMUR method to analyze multi-condition single-cell data. For the more complete version in R, see https://github.com/const-ae/lemur.

# Disclaimer

This is an early alpha version of the code, that is insufficiently documented and has not been extensively tested. I hope to make the implementation more robust and then make it available on PyPI in the coming weeks. Until then, use it at your own risk! 



## Citation

> Ahlmann-Eltze C, Huber W ({cite:year}`Ahlmann-Eltze2024`). 
“Analysis of multi-condition single-cell data with latent embedding multivariate regression.” bioRxiv. 
doi:10.1101/2023.03.06.531268.

# Getting started

## Installation

You need to have Python 3.9 or newer installed on your system. 
There are several alternative options to install pyLemur:

<!--
1) Install the latest release of `pyLemur` from `PyPI <https://pypi.org/project/pyLemur/>`_:

```bash
pip install pyLemur
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/const-ae/pyLemur.git@main
```

## Documentation

For more information on the functions see the [API docs](https://pyLemur.readthedocs.io/page/api.html).

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].




[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/const-ae/pyLemur/issues