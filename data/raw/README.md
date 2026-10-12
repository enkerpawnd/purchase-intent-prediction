# Source dataset

`online_shoppers_intention.csv` is the unchanged CSV supplied to the team. Keep
this file read-only in project workflows; write derived files to
`data/processed/` instead. It contains 12,330 sessions and 18 columns (17
predictors plus the Boolean target `Revenue`).

## Attribution and license

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*
[Dataset]. UCI Machine Learning Repository.
[https://doi.org/10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

- Source and feature descriptions: [UCI dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Modifications: none; the CSV is copied byte-for-byte from the team-supplied source.
- When redistributing the CSV, retain this attribution and license link, and
  identify any future modifications.

Expected SHA-256 (the original CSV bytes, not the ZIP archive):

```text
b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9
```

To check it manually, run from the repository root:

```bash
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('data/raw/online_shoppers_intention.csv').read_bytes()).hexdigest())"
```

If the file is absent, restore it from the repository or download the ZIP using
the UCI page's **Download** button, extract `online_shoppers_intention.csv` into
this directory and compare its checksum. Do not silently substitute a changed
CSV if its checksum differs; record and investigate the difference first.
Git attributes disable line-ending conversion for this file so that checkout
on Windows preserves its bytes.

The detailed schema and data-quality audit belong to PIP-02; loading the CSV
does not establish feature availability or an evaluation protocol.
