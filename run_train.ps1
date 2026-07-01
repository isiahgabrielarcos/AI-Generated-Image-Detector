<#
.SYNOPSIS
    Launch training with AMD CPU performance fixes applied.

.DESCRIPTION
    Sets MKL_DEBUG_CPU_TYPE=5 in the process environment BEFORE Python
    starts.  This is critical: Intel MKL reads the CPU model at library
    load time (i.e. on "import torch"), so the variable must already exist
    in the environment — setting it inside Python via os.environ is too late.

    MKL_DEBUG_CPU_TYPE=5 forces MKL to treat the CPU as an Intel processor,
    enabling AVX2-optimised GEMM / BLAS code paths that Intel otherwise skips
    on AMD hardware.  On a Ryzen this can give a 2-4x BLAS speedup.

    Any extra arguments are forwarded to train.py, e.g.:
        .\run_train.ps1 --resume checkpoints/epoch_10.pt
        .\run_train.ps1 --config configs/default.yaml

.EXAMPLE
    # Normal training run
    .\run_train.ps1

    # Resume from a checkpoint
    .\run_train.ps1 --resume checkpoints/best_model.pt
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

# ── Apply MKL AMD fix ────────────────────────────────────────────────
$env:MKL_DEBUG_CPU_TYPE = "5"
Write-Host "[run_train] MKL_DEBUG_CPU_TYPE=5  (AMD AVX2 BLAS paths enabled)" `
    -ForegroundColor Cyan

# ── Launch training ──────────────────────────────────────────────────
if ($TrainArgs) {
    python train.py @TrainArgs
} else {
    python train.py
}
