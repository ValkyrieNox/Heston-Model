# Paper source — FinFlow Heston world model

Top-conference-style write-up of the project (two-stage Flow-Matching world model
+ strictly-proper path-distribution fine-tuning + few-step distillation).

## Files
- `main.tex` — the paper, English (Introduction, Related Work, Problem Setup,
  Method, Experiments & Results, Ablation, Comparative Analysis, Conclusion). No
  appendix; all results are in the main text.
- `main_zh.tex` — Chinese version (same content/numbers/tables), uses `ctexart`.
- `references.bib` — bibliography (method sources + baseline sources), shared.

## Compile — English (`main.tex`)
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
Requires a standard TeX Live (packages: amsmath, amssymb, booktabs, multirow,
natbib, hyperref, authblk, caption, xcolor, geometry).

## Compile — Chinese (`main_zh.tex`), must use XeLaTeX
```bash
xelatex main_zh
bibtex  main_zh
xelatex main_zh
xelatex main_zh
```
Requires the `ctex` package and a CJK font (TeX Live full / CTeX). `pdflatex` will
NOT work for the Chinese version.

## Data provenance
Every number in Tables 1–4 is read from the evaluation JSONs on host 403 under
`runs/experiments/p3_full_parallel/eval_*/evaluation/*.json`
(`pricing_fake_vs_mc_oracle.rmse_overall` for raw/cal pricing RMSE,
`stylized_facts_comparison.kurtosis_fake` for kurtosis). The two econometric
baselines (GARCH(1,1)-t, moving-block bootstrap) were generated and evaluated for
this write-up via `baseline_generate.py` (CPU only) under `eval_baselines_0603/`.
Pricing floor (real test vs MC oracle) = 0.165; real kurtosis = 4.60.
