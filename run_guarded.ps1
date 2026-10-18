#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$BaseModel,
    [Parameter(Mandatory)][ValidateSet('domain_train.py','sft_lora.py','sft_lora_v2.py','sft_lora_v3.py','test_specialized.py')][string]$Script,
    [ValidateSet('v1','v2','v3')][string]$Version,
    [int]$TimeoutMinutes = 90
)
$ErrorActionPreference = 'Stop'
function Read-Hardware {
    $ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $values = (& nvidia-smi --id=0 --query-gpu=temperature.gpu,memory.free --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'GPU monitoring failed' }
    $parts = $values.Trim().Split(',')
    [pscustomobject]@{ utc=(Get-Date).ToUniversalTime().ToString('o'); ram_gib=[math]::Round($ram,3); gpu_c=[double]$parts[0]; vram_mib=[double]$parts[1] }
}
function Quote-Argument([string]$value) { '"' + $value.Replace('"','\"') + '"' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$BaseModel = (Resolve-Path -LiteralPath $BaseModel).Path
$scriptPath = Join-Path $PSScriptRoot $Script
if ($Script -eq 'test_specialized.py' -and -not $Version) { throw 'Test exports need -Version v1, v2 or v3' }
$before = Read-Hardware
if ($before.ram_gib -lt 4.5 -or $before.vram_mib -lt 4500 -or $before.gpu_c -ge 75) {
    throw ('Startup resource limit: ' + ($before | ConvertTo-Json -Compress))
}
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,6)
$logDir = Join-Path $PSScriptRoot "logs/$stamp"
$null = New-Item -ItemType Directory -Path $logDir
$settings = @{OMP_NUM_THREADS='2';MKL_NUM_THREADS='2';OPENBLAS_NUM_THREADS='2';TOKENIZERS_PARALLELISM='false';HF_HUB_OFFLINE='1';CUDA_VISIBLE_DEVICES='0';PYTHONUNBUFFERED='1'}
$previous = @{}
$worker = $null
$exitCode = 1
$failure = $null
$start = Get-Date
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
        [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process')
    }
    $arguments = @('-u', (Quote-Argument $scriptPath), '--base-model', (Quote-Argument $BaseModel))
    if ($Version) { $arguments += @('--version',$Version) }
    $worker = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log')
    Write-Output "Started $Script $Version; PID $($worker.Id); logs $logDir"
    $lastReport = $start
    while (-not $worker.HasExited) {
        $sample = Read-Hardware
        $sample | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $logDir 'hardware.jsonl')
        if ($sample.ram_gib -lt 1.25 -or $sample.gpu_c -ge 80) { throw 'Runtime memory or GPU temperature limit reached' }
        if (((Get-Date)-$start).TotalMinutes -gt $TimeoutMinutes) { throw 'Run exceeded its time limit' }
        if (((Get-Date)-$lastReport).TotalSeconds -ge 30) {
            Write-Output ("Monitoring: free RAM {0:N2} GiB; GPU {1} C" -f $sample.ram_gib,$sample.gpu_c)
            $lastReport = Get-Date
        }
        Start-Sleep -Seconds 2
        $worker.Refresh()
    }
    $worker.WaitForExit()
    $exitCode = $worker.ExitCode
    if ($exitCode -ne 0) { throw "Worker exited with code $exitCode" }
}
catch {
    $failure = $_.Exception.Message
    Write-Output "Stopped: $failure"
}
finally {
    if ($worker -and -not $worker.HasExited) {
        $worker.Kill($true)
        $worker.WaitForExit()
    }
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    @{script=$Script;version=$Version;started_utc=$start.ToUniversalTime().ToString('o');finished_utc=(Get-Date).ToUniversalTime().ToString('o');exit_code=$exitCode;failure=$failure} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'run.json')
    Write-Output "Finished; logs $logDir"
}
if ($failure) { exit 1 }
exit $exitCode
