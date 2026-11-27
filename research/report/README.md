# Agentic VI — living research report

Current version: **0.1 (27 November 2026)**, covering Tasks 001–003.

The authoritative editable report is [agentic_vi_report.tex](agentic_vi_report.tex). It is self-contained, including the TikZ radar chart, and needs no external figure files. Compile it with a standard TeX Live or MacTeX installation:

```sh
cd research/report
latexmk -pdf agentic_vi_report.tex
```

Alternatively, run `pdflatex agentic_vi_report.tex` twice. The resulting PDF is six pages in version 0.1. The compiled PDF is a derived artifact; source is tracked here.

## Maintenance

- Update the version, date, and scope when reviewed research milestones are added.
- Keep mathematical definitions, data provenance, computational accounting, experimental findings, and limitations together.
- Preserve the frozen protocols and original task reports; do not replace raw evidence with the summary.
- The radar chart is descriptive and uses axis-specific visual scales, not calibrated difficulty scores. Its data come from Task 003's archived per-fit results.
- Record future revisions as separate commits for human review. No new experiments are authorized by editing this report.

[Task 001](../task001/REPORT.md) · [Task 002](../task002/REPORT.md) · [Task 003](../task003/REPORT.md) · [VDP specification](../../VDP.md)
