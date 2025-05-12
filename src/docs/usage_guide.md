
# Mobile-Optimized 3proxy Usage Guide

## Overview

This is a mobile-optimized deployment of 3proxy, a tiny proxy server. The configuration has been specially tuned for mobile connections, with reduced bandwidth consumption and improved latency.

## Available Services

| Service | Port | Description |
|---------|------|-------------|
| SOCKS5  | 5000 | SOCKS proxy with TCP optimization for mobile |
| HTTP(S) | 8080 | HTTP proxy with compression support |
| DNS     | 5353 | Low-bandwidth DNS caching proxy |
| TCP Map | 2222 | SSH access through firewall (example) |
| UDP Map | 5400 | Voice/video calls through firewall |

## Authentication

Default credentials:
- Username: `admin`
- Password: `mobileproxy2024`

## Mobile Client Setup

### Android

1. **SOCKS Proxy**:
   - Install "Proxy Settings" app
   - Set host to your Replit URL
   - Set port to 5000
   - Enable authentication and enter credentials

2. **HTTP Proxy**:
   - Go to WiFi settings
   - Long press your connection
   - Select "Modify network"
   - Set proxy to "Manual"
   - Enter host and port 8080
   - Enter username and password

### iOS

1. **HTTP Proxy**:
   - Go to Settings → Wi-Fi
   - Tap the (i) icon next to your network
   - Scroll down to "HTTP Proxy"
   - Set to "Manual"
   - Enter server and port 8080
   - Enable Authentication
   - Enter username and password

## Bandwidth Optimization Tips

- Use the DNS proxy (port 5353) to reduce DNS lookup latency
- Enable compression in your HTTP clients when available
- For mobile web browsing, the HTTP proxy (8080) is more efficient than SOCKS
- Use the TCP/UDP port mapping for specific applications to bypass app-level proxy settings

## Security Notes

- Change the default password in `/etc/3proxy/passwd`
- Enable SSL for sensitive connections
- The proxy logs all connections by default in `/logs/`
