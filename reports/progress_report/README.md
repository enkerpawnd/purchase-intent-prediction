# Build the progress report

From this directory, run:

```bash
make pdf
```

Or run the equivalent command directly:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

The result is `main.pdf`. This uses the installed MacTeX/TeX Live tools.
`latexmk` runs LaTeX, BibTeX and the subsequent LaTeX passes needed to resolve
citations and section, table and figure references. Running `pdflatex` just once
leaves question marks and may omit the bibliography.

If an editor offers build recipes, select a **latexmk PDF** recipe. An equivalent
manual sequence is `pdflatex main`, `bibtex main`, `pdflatex main`, and
`pdflatex main`, all run from this directory.
