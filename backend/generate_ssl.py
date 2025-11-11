#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for Eden API HTTPS
"""

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta
import os

# Create SSL directory
ssl_dir = os.path.join(os.path.dirname(__file__), 'ssl')
os.makedirs(ssl_dir, exist_ok=True)

print("Generating SSL certificate for Eden API...")

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Generate certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Eden Trading Bot"),
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
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
        x509.DNSName("localhost"),
        x509.DNSName("127.0.0.1"),
        x509.DNSName("*.compute.amazonaws.com"),
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Write private key
key_path = os.path.join(ssl_dir, 'key.pem')
with open(key_path, "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

# Write certificate
cert_path = os.path.join(ssl_dir, 'cert.pem')
with open(cert_path, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print(f"✓ SSL certificate generated successfully!")
print(f"  Certificate: {cert_path}")
print(f"  Private Key: {key_path}")
print(f"  Valid for: 5 years")
print()
print("Backend will now run on HTTPS!")
