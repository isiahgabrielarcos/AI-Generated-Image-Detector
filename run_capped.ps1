<#
run_capped.ps1
──────────────
Run any training/eval command with hard CPU and RAM ceilings so an unattended
(overnight) run can't lock up the PC.

CPU cap : pins the process to a fraction of logical cores (ProcessorAffinity)
          + lowers priority, so the rest of the machine stays responsive.
RAM cap : a watchdog polls system memory; if usage stays above the kill
          threshold it terminates the run (recoverable — training checkpoints
          every epoch; relaunch with --resume).

USAGE  (pass the python command as ONE quoted string via -Cmd)
─────
  # 1e-15 demo, capped at 70% CPU / kill if RAM > 88%
  .\run_capped.ps1 -Cmd "train.py --config configs/lr_1e-15.yaml"

  # custom caps
  .\run_capped.ps1 -CpuPercent 60 -RamKillPercent 85 -Cmd "train.py --config configs/lr_1e-7.yaml"

  # works for any script (finetune, evaluate, ...)
  .\run_capped.ps1 -Cmd "finetune.py --checkpoint checkpoints/pergen_split_best_model.pt --target_generators DFDC"

The -Cmd string is split on spaces and passed to .venv\Scripts\python.exe.
(Individual arguments must not contain spaces — true for all paths here.)
#>

[CmdletBinding()]
param(
    [int]$CpuPercent      = 70,     # max % of logical cores to use
    [int]$RamWarnPercent  = 75,     # print a warning above this system RAM use
    [int]$RamKillPercent  = 88,     # kill the run if system RAM use stays above this
    [int]$KillSustainSec  = 20,     # must exceed kill threshold this long before killing
    [int]$PollSec         = 5,      # how often to check RAM
    [ValidateSet("Idle","BelowNormal","Normal")]
    [string]$Priority     = "BelowNormal",
    [Parameter(Mandatory=$true)]
    [string]$Cmd
)

$ErrorActionPreference = "Stop"

# Tokenise the command string into individual python arguments
$PyArgs = $Cmd -split '\s+' | Where-Object { $_ -ne '' }
if (-not $PyArgs -or $PyArgs.Count -eq 0) { throw "Empty -Cmd. Example: .\run_capped.ps1 -Cmd 'train.py --config configs/lr_1e-15.yaml'" }

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "python not found at $python" }

# ── CPU sizing ────────────────────────────────────────────────────────
$logical  = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
$useCores = [Math]::Max(1, [Math]::Floor($logical * $CpuPercent / 100.0))
$mask     = ([int64]1 -shl $useCores) - 1     # lowest $useCores bits set

# Thread caps (read by train.py via AIGC_MAX_THREADS; others for safety)
$env:AIGC_MAX_THREADS   = "$useCores"
$env:OMP_NUM_THREADS    = "$useCores"
$env:MKL_NUM_THREADS    = "$useCores"
$env:OPENBLAS_NUM_THREADS = "$useCores"
$env:NUMEXPR_NUM_THREADS  = "$useCores"

$totalRamGB = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 1)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  run_capped" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ("  CPU      : {0}/{1} logical cores ({2}%), priority {3}" -f $useCores,$logical,$CpuPercent,$Priority)
Write-Host ("  RAM      : {0} GB total | warn >{1}% | KILL >{2}% sustained {3}s" -f $totalRamGB,$RamWarnPercent,$RamKillPercent,$KillSustainSec)
Write-Host ("  Command  : python {0}" -f ($PyArgs -join ' '))
Write-Host "------------------------------------------------------------"

# ── Launch ────────────────────────────────────────────────────────────
$p = Start-Process -FilePath $python -ArgumentList $PyArgs -PassThru -NoNewWindow
Start-Sleep -Milliseconds 500
try {
    $p.ProcessorAffinity = [IntPtr]$mask
    $p.PriorityClass     = [System.Diagnostics.ProcessPriorityClass]::$Priority
    Write-Host ("  PID {0} launched, affinity=0x{1:X}, priority={2}" -f $p.Id,$mask,$Priority) -ForegroundColor Green
} catch {
    Write-Warning "Could not set affinity/priority: $_"
}

# ── RAM watchdog ──────────────────────────────────────────────────────
$os         = Get-CimInstance Win32_OperatingSystem
$totalKB    = $os.TotalVisibleMemorySize
$overSince  = $null

while (-not $p.HasExited) {
    Start-Sleep -Seconds $PollSec
    if ($p.HasExited) { break }

    $os       = Get-CimInstance Win32_OperatingSystem
    $freeKB   = $os.FreePhysicalMemory
    $usedPct  = [Math]::Round((($totalKB - $freeKB) / $totalKB) * 100, 1)
    $usedGB   = [Math]::Round((($totalKB - $freeKB) / 1MB), 1)

    if ($usedPct -ge $RamKillPercent) {
        if (-not $overSince) { $overSince = Get-Date }
        $elapsed = (New-TimeSpan -Start $overSince -End (Get-Date)).TotalSeconds
        Write-Host ("  [watchdog] RAM {0}% ({1} GB) ABOVE kill line - {2:N0}/{3}s" -f $usedPct,$usedGB,$elapsed,$KillSustainSec) -ForegroundColor Red
        if ($elapsed -ge $KillSustainSec) {
            Write-Host "  [watchdog] KILLING run to protect the PC. Relaunch with --resume to continue." -ForegroundColor Red
            try { Stop-Process -Id $p.Id -Force } catch {}
            break
        }
    }
    elseif ($usedPct -ge $RamWarnPercent) {
        $overSince = $null
        Write-Host ("  [watchdog] RAM {0}% ({1} GB) - above warn line, watching" -f $usedPct,$usedGB) -ForegroundColor Yellow
    }
    else {
        $overSince = $null
    }
}

if ($p.HasExited) {
    Write-Host ("------------------------------------------------------------")
    Write-Host ("  Process exited with code {0}" -f $p.ExitCode)
}
