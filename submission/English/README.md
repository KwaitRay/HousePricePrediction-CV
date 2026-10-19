# COMP90086 submission materials

Shifeng Chen — 1686631; Haoxuan Ying — 1255329.

- **report.pdf**: English IEEE report, four A4 content pages plus references on page five.
- **code.zip**: separate Canvas code upload (also expanded under `code/` for review).
- **code/predictions.csv**: unchanged 3000-row Kaggle submission, public MSE 47291.34608 (manual feedback).
- **report_source/**: LaTeX, references and figures. Compile `report_draft.tex` with IEEEtran/pdflatex twice or Tectonic.
- **code/README.txt**: dependencies, expected dataset layout, experiment entry points and reproduction scope.

Canvas requires the report and code as separate uploads. The combined English archive
is for transfer/review. Trained model heads, pretrained weights, feature caches and
supplied images are omitted. All experiment source/configuration code discussed in
the report is included. `code/reproduce_final.py` provides staged fixed-recipe
reproduction without historical fitted heads or run IDs; original experiment
scripts retain execution guards. See `code/README.txt` for commands and checks.

The report and prediction files are the current snapshot. Confirm the Kaggle team
name contains both student IDs and complete the course's separate peer assessment.
