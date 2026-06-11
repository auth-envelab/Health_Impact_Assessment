# Reproduction Workflow

The analysis workflow starts from the controlled harmonized dataset and regenerates manuscript tables, supplementary tables, aggregate results, and validation reports. Final approved figures are included as manuscript output artifacts, with aggregate outputs and validation reports provided to support reproducibility of the underlying results. The workflow does not process raw or unharmonized data.

Participant-level data are not included in this repository. The controlled harmonized dataset remains available only through the approved controlled-data environment.

## Windows PowerShell Run

```powershell
$Repo = "<PATH_TO_REPOSITORY>"
$HarmonizedZip = "<PATH_TO_CONTROLLED_HARMONIZED_DATASET_ZIP>"
Set-Location $Repo

.\Run-Manuscript-Reproducibility.ps1 `
  -HarmonizedZip "$HarmonizedZip" `
  -OutDir "<OUTPUT_DIR>" `
  -NSamples 10000 `
  -BuildPublicCandidate
```

## Controlled-Data Environment

Run the workflow inside the approved controlled-data environment with access to the controlled harmonized dataset. All logs and run outputs are written under the selected output directory.

## Public Aggregate Mode

Public aggregate mode uses only aggregate outputs included in a validated public candidate.

```powershell
.\Run-Manuscript-Reproducibility.ps1 `
  -OutDir "<OUTPUT_DIR>" `
  -PublicAggregateOnly `
  -CandidateDir "<PATH_TO_PUBLIC_AGGREGATE_CANDIDATE>"
```

## Outputs

The workflow writes final approved figure artifacts, tables, aggregate CSV files, validation reports, and an optional public aggregate candidate under the selected output directory.

## Validation Reports

Validation reports check denominator consistency, HIA and YLL row counts, model support counts, manuscript item presence, figure template fidelity, absence of unsupported p-value formatting, and public candidate safety.
