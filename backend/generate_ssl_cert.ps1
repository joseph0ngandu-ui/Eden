# Generate Self-Signed SSL Certificate for Eden API

$certPath = "C:\Users\Administrator\Eden\backend\ssl"

# Create SSL directory
New-Item -ItemType Directory -Force -Path $certPath | Out-Null

Write-Host "Generating SSL certificate for Eden API..." -ForegroundColor Cyan

# Generate self-signed certificate
$cert = New-SelfSignedCertificate `
    -Subject "CN=Eden Trading Bot API" `
    -DnsName "localhost", "127.0.0.1", "*.compute.amazonaws.com" `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -NotAfter (Get-Date).AddYears(5) `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -FriendlyName "Eden API Certificate" `
    -HashAlgorithm SHA256 `
    -KeyUsage DigitalSignature, KeyEncipherment `
    -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.1")

Write-Host "Certificate created with thumbprint: $($cert.Thumbprint)" -ForegroundColor Green

# Export certificate to PFX
$pfxPassword = ConvertTo-SecureString -String "eden2025" -Force -AsPlainText
$pfxPath = Join-Path $certPath "eden-api.pfx"
Export-PfxCertificate -Cert $cert -FilePath $pfxPath -Password $pfxPassword | Out-Null

# Export to PEM format for uvicorn
$certPemPath = Join-Path $certPath "cert.pem"
$keyPemPath = Join-Path $certPath "key.pem"

# Use OpenSSL if available, otherwise use .NET
try {
    # Convert PFX to PEM using OpenSSL
    & openssl pkcs12 -in $pfxPath -out $certPemPath -clcerts -nokeys -passin pass:eden2025 2>$null
    & openssl pkcs12 -in $pfxPath -out $keyPemPath -nocerts -nodes -passin pass:eden2025 2>$null
    Write-Host "✓ Certificates exported to PEM format" -ForegroundColor Green
} catch {
    # Fallback: Export cert manually
    $certBytes = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
    $certPem = "-----BEGIN CERTIFICATE-----`n"
    $certPem += [System.Convert]::ToBase64String($certBytes, [System.Base64FormattingOptions]::InsertLineBreaks)
    $certPem += "`n-----END CERTIFICATE-----`n"
    Set-Content -Path $certPemPath -Value $certPem
    
    Write-Host "✓ Certificate exported (key requires manual extraction)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "SSL Certificate Location:" -ForegroundColor Cyan
Write-Host "  PFX: $pfxPath" -ForegroundColor White
Write-Host "  Cert: $certPemPath" -ForegroundColor White
Write-Host "  Key: $keyPemPath" -ForegroundColor White
Write-Host "  Password: eden2025" -ForegroundColor White
Write-Host ""
Write-Host "✓ SSL certificate ready for HTTPS!" -ForegroundColor Green
