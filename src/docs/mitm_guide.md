# Man-in-the-Middle Configuration Guide

This guide explains how to set up the 3proxy SSL Plugin for intercepting HTTPS traffic.

**Warning**: Using these techniques without proper authorization may be illegal in many jurisdictions.

## Requirements

- 3proxy compiled with SSL support
- OpenSSL for certificate generation
- Root certificate installed on client devices

## Setup

1. Generate a CA certificate and key:

```bash
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/CN=MobileProxy CA"
```

2. Add the following to your 3proxy.cfg:

```
plugin /plugins/SSLPlugin.so ssl_plugin
ssl_mitm
ssl_certcache /certs/cache/
ssl_server_ca_file /certs/ca.crt
ssl_server_ca_key /certs/ca.key
```

3. Restart 3proxy

## Client Setup

### Android

1. Download the ca.crt file to your device
2. Go to Settings → Security → Encryption & Credentials
3. Tap "Install a certificate" → CA Certificate
4. Select the downloaded ca.crt file
5. Follow the prompts to install

### iOS

1. Email the ca.crt file to yourself
2. Open the attachment on your iOS device
3. Go to Settings → Profile Downloaded
4. Tap "Install" and follow the prompts
5. Go to Settings → General → About → Certificate Trust Settings
6. Enable full trust for the installed certificate

## Testing

1. Configure your device to use the proxy (port 8443)
2. Visit https://example.com in your browser
3. Check the certificate details - it should be signed by your CA

## Analyzing Traffic

For mobile traffic analysis:

1. Enable detailed logging in 3proxy.cfg:
```
log /logs/ssl-%y%m%d.log D
logformat "L%C - %U [%t] \"%T\" %E %I %h %T %{User-Agent}"
```

2. Use grep to filter mobile-specific traffic:
```bash
grep -i "android\|iphone\|mobile" /logs/ssl-*.log