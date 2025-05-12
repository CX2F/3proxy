# Mobile-Optimized 3Proxy Usage Guide

This guide will help you set up and use the mobile-optimized version of 3Proxy.

## Available Services

- **HTTP Proxy**: Port 8080
- **SOCKS5 Proxy**: Port 1080
- **HTTPS Proxy with SSL**: Port 8443
- **Tor Proxy**: Port 9050

## Mobile Client Configuration

### Android

1. Go to Settings → Wi-Fi
2. Long-press your connected network
3. Select "Modify Network"
4. Check "Show advanced options"
5. Change Proxy settings to "Manual"
6. Enter the server IP and port 8080
7. Save the configuration

### iOS

1. Go to Settings → Wi-Fi
2. Tap the (i) icon next to your connected network
3. Scroll down to "HTTP Proxy" and select "Manual"
4. Enter the server IP and port 8080
5. Leave Authentication blank or enter admin/mobileproxy if required

### Mobile Apps

For apps that don't use system proxy settings:

1. Install a VPN app like "HTTP Injector" or "Proxifier"
2. Configure the VPN to use our SOCKS5 proxy (port 1080)
3. Enable the VPN to route all traffic through our proxy

## Optimizations

This 3proxy build includes:

- TCP optimization for mobile connections
- Buffer size optimization (16KB for mobile)
- Fast DNS lookups with caching
- SSL support for encrypted connections
- Tor integration for anonymous browsing

## Troubleshooting

If you experience connection issues:

1. Check that the proxy server is running
2. Verify the correct IP and port in your configuration
3. Try different ports (8080, 1080, 8443)
4. Make sure the proxy is reachable from your network