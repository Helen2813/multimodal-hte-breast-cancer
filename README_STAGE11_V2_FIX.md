# Stage 11 v2 — finish Stage 40 without tabulate

Stages 37–39 completed successfully and are not rerun.

`pandas.DataFrame.to_markdown()` required the optional `tabulate` package.
Stage 40 now uses an internal Markdown renderer. The preflight parses the
Python AST and confirms that no real `to_markdown` call or `tabulate` import
remains.

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage11_finish_stage40.ps1
```
