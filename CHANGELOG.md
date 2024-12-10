# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

# [0.3.1]

- Fix documentation of `cond()` return type and handle pd.Series in
  `model.predict()` (#5, thanks Mark Keller)

## [0.3.0]

- Depend on `formulaic_contrast` package
- Refactor `cond()` implementation to use `formulaic_contrast` implementation.

## [0.2.2]

- Sync with cookiecutter-template update (version 0.4)
- Bump required Python version to `3.10`
- Allow data frames as design matrices
- Allow matrices as input to LEMUR()

## [0.2.1]

- Change example gene to one with clearer differential expression pattern
- Remove error output in `align_harmony

## [0.2.0]

Major rewrite of the API. Instead of adding coefficients as custom fields
to the input `AnnData` object, the API now follows an object-oriented style
similar to scikit-learn or `SCVI`. This change was motivated by the feedback
during the submission to the `scverse` ecosystem.
([Thanks](<(https://github.com/scverse/ecosystem-packages/pull/156#issuecomment-2014676654)>) Gregor).

### Changed

- Instead of calling `fit = pylemur.tl.lemur(adata, ...)`, you now create a LEMUR model
  (`model = pylemur.tl.LEMUR(adata, ...)`) and subsequently call `model.fit()`, `model.align_with_harmony()`,
  and `model.predict()`.

## [0.1.0] - 2024-03-21

- Initial beta release of `pyLemur`
