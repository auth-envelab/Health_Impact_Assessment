# Controlled-Data Execution

Run the analysis workflow inside the controlled-data environment with the controlled harmonized dataset mounted at a private location.

Use placeholders in shared configuration files and supply private paths only at run time. Do not place private paths in public candidate files.

Expected command pattern:

```powershell
.\Run-Manuscript-Reproducibility.ps1 `
  -HarmonizedZip "<PATH_TO_CONTROLLED_HARMONIZED_DATASET_ZIP>" `
  -OutDir "analysis_outputs" `
  -NSamples 10000 `
  -BuildPublicCandidate
```

The workflow writes local run outputs under `analysis_outputs`. The public aggregate candidate includes only safe aggregate outputs, regenerated manuscript figures, tables, validation reports, descriptive scripts, placeholder configs, and checksums.
