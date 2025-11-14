$ErrorActionPreference = "Stop"

try {
    $domain = "edenbot.duckdns.org"
    $installPath = "C:\Tools\win-acme"
    $certOutputPath = "C:\EdenCerts"
    $zipPath = Join-Path $installPath "wacs.zip"

    # Ensure TLS 1.2 for GitHub/API
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    # Create directories if they don't exist
    New-Item -ItemType Directory -Force -Path $installPath | Out-Null
    New-Item -ItemType Directory -Force -Path $certOutputPath | Out-Null

    # Get latest win-acme Windows x64 release info from GitHub
    $githubHeaders = @{ 'User-Agent' = 'EdenSSLSetupScript/1.0' }
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/win-acme/win-acme/releases/latest" -Headers $githubHeaders
    if (-not $release -or -not $release.assets) {
        throw "FATAL: Unable to retrieve win-acme release information from GitHub."
    }

    $asset = $release.assets | Where-Object { $_.name -like "win-acme.v*.x64.pluggable.zip" } | Select-Object -First 1
    if (-not $asset) {
        throw "FATAL: No suitable win-acme Windows x64 ZIP asset found in latest release."
    }

    $downloadUrl = $asset.browser_download_url
    if (-not $downloadUrl) {
        throw "FATAL: win-acme asset has no download URL."
    }

    # Download win-acme ZIP
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing -Headers $githubHeaders

    # Extract win-acme
    Expand-Archive -Path $zipPath -DestinationPath $installPath -Force

    # Locate wacs.exe
    $wacsPath = Join-Path $installPath "wacs.exe"
    if (-not (Test-Path $wacsPath)) {
        # Some archives may extract into a subfolder; search for wacs.exe
        $wacsPath = Get-ChildItem -Path $installPath -Recurse -Filter "wacs.exe" | Select-Object -First 1 -ExpandProperty FullName
        if (-not $wacsPath) {
            throw "FATAL: wacs.exe not found after extraction."
        }
    }

    # Run win-acme to issue certificate (fully unattended)
    & $wacsPath `
        --target manual `
        --host $domain `
        --validation selfhosting `
        --validationmode http-01 `
        --store pemfiles `
        --pemfilespath $certOutputPath `
        --installation none `
        --accepttos `
        --verbose

    if ($LASTEXITCODE -ne 0) {
        throw "FATAL: win-acme certificate request failed with exit code $LASTEXITCODE."
    }

    # Verify cert files
    $fullchain = Join-Path $certOutputPath "fullchain.pem"
    $key = Join-Path $certOutputPath "key.pem"

    if (-not (Test-Path $fullchain)) {
        throw "FATAL: Missing certificate file: $fullchain"
    }
    if (-not (Test-Path $key)) {
        throw "FATAL: Missing key file: $key"
    }

    # Trigger renewal task handling
    & $wacsPath --renew
    if ($LASTEXITCODE -ne 0) {
        throw "FATAL: win-acme renew command failed with exit code $LASTEXITCODE."
    }

    Write-Output "SSL setup complete."
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
