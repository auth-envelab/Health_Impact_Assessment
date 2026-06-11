param(
  [string]$HarmonizedZip,
  [string]$OutDir = "local_outputs\manuscript_reproducibility\manual_run",
  [int]$NSamples = 10000,
  [switch]$BuildPublicCandidate,
  [switch]$PublicAggregateOnly,
  [string]$CandidateDir = ""
)

$ErrorActionPreference = "Stop"

function Write-StepMessage {
  param([string]$Message)
  Write-Host "[analysis workflow] $Message"
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcPath = Join-Path $RepoRoot "src"

if (!(Test-Path $SrcPath)) {
  Write-Error "[analysis workflow] Missing src directory: $SrcPath"
  exit 1
}

$existingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
  $env:PYTHONPATH = "$SrcPath;$RepoRoot"
} else {
  $env:PYTHONPATH = "$SrcPath;$RepoRoot;$existingPythonPath"
}

Set-Location $RepoRoot
$LogDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "manuscript_reproducibility_runner.log"

Write-StepMessage "Checking Python availability."
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
  Write-Error "Python was not found on PATH."
  exit 2
}

if ($PublicAggregateOnly) {
  if ([string]::IsNullOrWhiteSpace($CandidateDir)) {
    $CandidateDir = $RepoRoot
  }
  if (-not (Test-Path -LiteralPath $CandidateDir)) {
    Write-Error "Candidate directory was not found."
    exit 2
  }
} else {
  if ([string]::IsNullOrWhiteSpace($HarmonizedZip)) {
    Write-Error "-HarmonizedZip is required unless -PublicAggregateOnly is used."
    exit 2
  }
  if (-not (Test-Path -LiteralPath $HarmonizedZip)) {
    Write-Error "Harmonized dataset archive was not found."
    exit 2
  }
}

$ArgsList = @(
  "scripts\run_manuscript_reproducibility_from_harmonized.py",
  "--out-dir", $OutDir,
  "--n-samples", $NSamples.ToString()
)

if ($PublicAggregateOnly) {
  $ArgsList += "--public-aggregate-only"
  $ArgsList += @("--candidate-dir", $CandidateDir)
} else {
  $ArgsList += @("--harmonized-zip", $HarmonizedZip)
}

if ($BuildPublicCandidate) {
  $ArgsList += "--build-public-candidate"
}

Write-StepMessage "Starting manuscript reproducibility workflow."
Write-StepMessage "Output directory: $OutDir"
& python @ArgsList 2>&1 | Tee-Object -FilePath $LogFile
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
  Write-StepMessage "Workflow failed. See the log under local_outputs."
  exit $ExitCode
}

Write-StepMessage "Workflow completed. See output and validation reports under local_outputs."
exit 0
