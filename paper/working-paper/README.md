# Working paper: mathematical companion to `jump-diffusion-estimation`

Pedagogical working paper (English) deriving the mathematics behind every
component of the library and grounding each derivation in the actual code:
model + Bernoulli transition density, the five jump families and their
closed-form convolutions, the FFT convolution scheme, MLE with L-BFGS-B and
differential evolution, the three inference routes (profile / Wald /
bootstrap), the bootstrap jump test, and the goodness-of-fit comparison.

**Status**: working draft, versioned in the repository (decision of
2026-07-14) — LaTeX sources plus the compiled [main.pdf](main.pdf);
LaTeX build artifacts stay ignored. The methods-paper scoping notes in
`../` remain private.

## Structure

```
main.tex               # preamble, abstract, \input of sections
sections/01-...tex     # one file per section, numbered in reading order
references.bib         # BibTeX database
```

## Compiling

BasicTeX (TeX Live 2026) is installed locally (2026-07-12, via
`brew install --cask basictex`; binaries in `/Library/TeX/texbin`, on PATH
in new shells), plus `latexmk` (installed 2026-07-12 via
`sudo /Library/TeX/texbin/tlmgr install latexmk` — note `sudo` needs the
full path, its restricted PATH doesn't include texbin). From this
directory:

```bash
latexmk -pdf main      # runs pdflatex/bibtex as many times as needed
latexmk -c             # clean aux files (keeps the PDF)
```

Manual equivalent, if ever needed:
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Alternative: upload the folder to **Overleaf** — only standard packages
are used (amsmath, listings, natbib, hyperref, booktabs, xcolor,
microtype), so it compiles out of the box.

## Conventions

- Every listing is a trimmed excerpt of the *actual* library source —
  docstrings elided, logic never altered. If the code changes, the paper
  must change with it.
- "Design decision" environments record the *why* behind numerical
  constants and implementation choices (grid sizes, penalizations,
  optimizer defaults), with pointers to Ospina (2009) / Ardia et al. (2011)
  where they originate.

## Sync log

- 2026-07-14: updated for the 2026-07-13 audit fixes (v0.2.1 → v0.2.2
  line): FFT grid coverage guard + pointwise out-of-grid zeros (PR #70),
  `characteristic_location` / `_moment_grid` / generic `mean()`/
  `variance()` + exact `diagnostics()` increment moments (PR #71),
  `ValidationExperiment` NaN relative errors & `within_5pct_rate` rename
  (PR #72). Sections 3, 4, 6 and 8 touched; 28 pp.

## Items to verify before circulating

- [ ] Azzalini (1986) citation details (volume/pages) for the skew-normal
      convolution closure — flagged in `sections/03-distributions.tex`.
- [ ] Re-check every listing against the source after any library release.
