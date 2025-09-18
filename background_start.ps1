param()
$ErrorActionPreference = 'Continue'

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'python'
$script = Join-Path $root 'run_complete_optimization_bg.py'
$log    = Join-Path $root 'complete_opt.log'
$errlog = Join-Path $root 'complete_opt.err.log'
$pidFile = Join-Path $root 'complete_opt.pid'

"[$(Get-Date -Format s)] Launching Eden Complete MT5 optimization..." | Out-File -FilePath $log -Encoding utf8 -Force

try {
  # Force UTF-8 for Python stdio to avoid UnicodeEncodeError on emojis
  $env:PYTHONIOENCODING = 'utf-8'
  $env:PYTHONUTF8 = '1'

  $args = "-X utf8 `"$script`""
  $p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errlog
  $p.Id | Out-File -FilePath $pidFile -Encoding ascii -Force
  Start-Sleep -Milliseconds 200
  "[$(Get-Date -Format s)] Started PID $($p.Id). Logs: $log (stdout), $errlog (stderr)" | Out-File -FilePath ($log + '.meta') -Encoding utf8 -Force
}
catch {
  "[$(Get-Date -Format s)] Failed to start: $($_.Exception.Message)" | Add-Content -Path $log -Encoding utf8
  exit 1
}
