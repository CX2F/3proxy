
# Man-in-the-Middle (MITM) Configuration Guide

## Overview

This guide explains how to configure 3proxy as a MITM proxy to decrypt and inspect HTTPS traffic for debugging, content filtering, or security analysis.

⚠️ **IMPORTANT**: Using MITM techniques without proper authorization may violate privacy laws and terms of service. Use only in controlled environments with proper consent.

## Prerequisites

- OpenSSL (included in the setup)
- SSLPlugin for 3proxy (included in this build)

## Setup Instructions

### 1. Generate CA Certificate

```bash
# Generate CA private key
openssl genrsa -out /certs/ca.key 4096

# Generate CA certificate
openssl req -new -x509 -key /certs/ca.key -out /certs/ca.crt -days 3650 -subj "/CN=Mobile Proxy CA"
```

### 2. Configure SSLPlugin

Add these lines to your `3proxy.cfg` file:

```
# Load SSL plugin
plugin /plugins/SSLPlugin.so ssl_plugin

# Configure MITM settings
ssl_mitm
ssl_addca /certs/ca.crt
ssl_addhostname *.example.com

# Start MITM proxy
proxy -p8443
```

### 3. Install CA Certificate on Mobile Devices

#### Android:

1. Download the CA certificate to your device
2. Go to Settings → Security → Install from storage
3. Find and select the downloaded CA certificate
4. Name the certificate and select "VPN and apps" for credential use

#### iOS:

1. Email the CA certificate to yourself or host it on a website
2. Open the file on your iOS device
3. Go to Settings → Profile Downloaded
4. Install the CA certificate
5. Go to Settings → General → About → Certificate Trust Settings
6. Enable full trust for the CA certificate

## MITM Proxy Usage Examples

### Content Filtering

```
# Block certain domains
ssl_addhostname *.example.com,*.ads.com block

# Redirect sites
ssl_addhostname banking.example.com redirect https://safe.example.com
```

### Traffic Analysis

```
# Log all HTTPS traffic
ssl_logdump /logs/ssl_dump_%y%m%d.log
```

### Mobile App Debugging

```
# Intercept specific app traffic
ssl_addhostname api.example.com debug
```

## Troubleshooting

- **Certificate Warnings**: Ensure the CA certificate is properly installed on client devices
- **Connection Failures**: Some apps use certificate pinning and will reject MITM certificates
- **Performance Issues**: MITM proxying requires more CPU resources; reduce the number of concurrent connections for mobile devices
