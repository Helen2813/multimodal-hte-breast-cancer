# Stage 12 v2 runner newline fix

The prior PowerShell file contained literal `\n` characters instead of real
line breaks. PowerShell therefore parsed the entire runner as one line.

Copy `run_stage12_v2_pilot.ps1` into the project root and replace the existing
file.

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage12_v2_pilot.ps1
```

No analysis script ran during the failed parser attempt, so no checkpoint
cleanup is required.
