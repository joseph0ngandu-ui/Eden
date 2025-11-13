"""
Generate SSL Certificate for DuckDNS Domain
Creates a self-signed certificate for edenbot.duckdns.org
"""

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta
import os

# Certificate settings
DOMAIN = "edenbot.duckdns.org"
CERT_DIR = os.path.join(os.path.dirname(__file__), "ssl")
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "key.pem")

print(f"Generating SSL certificate for {DOMAIN}...")

# Ensure SSL directory exists
os.makedirs(CERT_DIR, exist_ok=True)

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Generate certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Cloud"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "AWS"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Eden Trading"),
    x509.NameAttribute(NameOID.COMMON_NAME, DOMAIN),
])

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.utcnow()
).not_valid_after(
    datetime.utcnow() + timedelta(days=365*5)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName(DOMAIN),
        x509.DNSName(f"*.{DOMAIN}"),  # Wildcard for subdomains
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Write certificate to file
with open(CERT_FILE, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

# Write private key to file
with open(KEY_FILE, "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

print(f"✓ SSL certificate generated successfully!")
print(f"  Certificate: {CERT_FILE}")
print(f"  Private Key: {KEY_FILE}")
print(f"  Domain: {DOMAIN}")
print(f"  Valid for: 5 years")
print("")
print("Backend will now run on HTTPS with edenbot.duckdns.org!")
