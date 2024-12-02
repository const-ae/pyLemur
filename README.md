# pyLemur

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/const-ae/pyLemur/test.yaml?branch=main
[link-tests]: https://github.com/const-ae/pyLemur/actions/workflows/test.yaml
[link-docs]: https://pyLemur.readthedocs.io
[badge-docs]: http://readthedocs.org/projects/pylemur/badge

The Python implementation of the LEMUR method to analyze multi-condition single-cell data. For the more complete version in R, see [github.com/const-ae/lemur](https://github.com/const-ae/lemur). To learn more check-out the [function documentation](https://pylemur.readthedocs.io/page/api.html) and the [tutorial](https://pylemur.readthedocs.io/page/notebooks/Tutorial.html) at [pylemur.readthedocs.io](https://pylemur.readthedocs.io). To check-out the source code or submit an issue go to [github.com/const-ae/pyLemur](https://github.com/const-ae/pyLemur)

## Citation

> Ahlmann-Eltze C, Huber W (2024).
> “Analysis of multi-condition single-cell data with latent embedding multivariate regression.” bioRxiv.
> [doi:10.1101/2023.03.06.531268](https://doi.org/10.1101/2023.03.06.531268).

# Getting started

## Installation

You need to have Python 3.10 or newer installed on your system.
There are several alternative options to install pyLemur:

Install the latest release of `pyLemur` from [PyPI](https://pypi.org/project/pyLemur/):

```bash
pip install pyLemur
```

Alternatively, install the latest development version directly from Github:

```bash
pip install git+https://github.com/const-ae/pyLemur.git@main
```

## Documentation

For more information on the functions see the [API docs](https://pyLemur.readthedocs.io/page/api.html) and the [tutorial](https://pylemur.readthedocs.io/page/notebooks/Tutorial.html).

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/const-ae/pyLemur/issues

## Building

Install the package in editable mode:

```
pip install ".[dev,doc,test]"
```

Build the documentation locally

```
cd docs
make html
open _build/html/index.html
```

Run the unit tests

```
pytest
```

Run pre-commit hooks manually

```
pre-commit run --all-files
```

or individually

```
ruff check
```
