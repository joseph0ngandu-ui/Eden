param()
$ErrorActionPreference = 'SilentlyContinue'

$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$pointer = Join-Path $root 'last_complete_results.json'
$flag    = Join-Path $root 'results_ready.flag'

function Show-Popup($msg, $title) {
  try {
    $wshell = New-Object -ComObject WScript.Shell
    # 0x0 OK button, 0x40 Information icon, 0x40000 Top-most
    $wshell.Popup($msg, 10, $title, 0x0 + 0x40 + 0x40000) | Out-Null
  } catch {}
}

while (-not (Test-Path $pointer)) {
  Start-Sleep -Seconds 20
}

try {
  $data = Get-Content -Path $pointer -Raw | ConvertFrom-Json
} catch {
  Start-Sleep -Seconds 5
  try { $data = Get-Content -Path $pointer -Raw | ConvertFrom-Json } catch { $data = $null }
}

if ($null -eq $data) { exit 0 }

$resultsFile = $data.results_file
$status      = $data.status

if ($status -eq 'success' -and $resultsFile -and (Test-Path $resultsFile)) {
  "[$(Get-Date -Format s)] Results ready: $resultsFile" | Out-File -FilePath $flag -Encoding utf8 -Force
  Show-Popup "Results JSON ready:\n$resultsFile" "Eden Complete"
  exit 0
}

# If not success yet, poll for results file existence if path is known
if ($resultsFile) {
  while (-not (Test-Path $resultsFile)) {
    Start-Sleep -Seconds 20
  }
  "[$(Get-Date -Format s)] Results ready (post): $resultsFile" | Out-File -FilePath $flag -Encoding utf8 -Force
  Show-Popup "Results JSON ready:\n$resultsFile" "Eden Complete"
} else {
  # Pointer indicated error/empty, notify accordingly
  $msg = "Status: $status"
  if ($data.error) { $msg += "`nError: $($data.error)" }
  Show-Popup $msg "Eden Complete"
}
