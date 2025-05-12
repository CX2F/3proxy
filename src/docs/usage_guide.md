
# 3proxy Full Network Configuration Guide

This guide explains how to use the various proxy services configured in this setup.

## Available Services

| Service | Port | Description |
|---------|------|-------------|
| SOCKS5 | 5000 | Secure SOCKS5 proxy with authentication |
| HTTP(S) | 8080 | HTTP and HTTPS proxy |
| DNS | 5353 | DNS proxy service |
| SMTP | 5025 | SMTP relay service |
| POP3 | 5110 | POP3 proxy service |
| TCP Forward | 8888 | Example TCP port forwarding (to port 80) |
| UDP Forward | 5353 | Example UDP port forwarding (to DNS) |
| Tor Proxy | 5555 | Traffic routed through Tor network |

## Authentication

Authentication is enabled for the SOCKS5 proxy. Use the following credentials:
- Username: `admin`
- Password: `password123`

## Usage Examples

### SOCKS5 Proxy

```bash
# Using curl with SOCKS5
curl --socks5 admin:password123@0.0.0.0:5000 http://example.com

# Configure Firefox to use SOCKS5
# Settings -> Network Settings -> Manual proxy configuration
# SOCKS Host: 0.0.0.0, Port: 5000
```

### HTTP Proxy

```bash
# Using curl with HTTP proxy
curl -x http://0.0.0.0:8080 http://example.com

# Using wget with proxy
wget -e use_proxy=yes -e http_proxy=0.0.0.0:8080 http://example.com
```

### DNS Proxy

```bash
# Query through DNS proxy
dig @0.0.0.0 -p 5353 example.com

# Configure system to use custom DNS
# Edit /etc/resolv.conf:
# nameserver 0.0.0.0
```

### Tor Integration

Traffic sent to port 5555 will be routed through the Tor network:

```bash
# Using curl with Tor proxy
curl -x http://0.0.0.0:5555 https://check.torproject.org/
```

## Custom TLD Configuration

This setup includes a custom TLD `mytld` for internal testing:

- `ns1.mytld` points to 127.0.0.1
- `www.mytld` points to 127.0.0.1
- `test.mytld` points to 127.0.0.1

To use custom TLDs, ensure your system is configured to use the DNS proxy (port 5353).

## Mobile Optimization

This configuration includes TCP optimizations specifically for mobile devices:
- TCP_NODELAY reduces latency
- TCP_QUICKACK improves response times
- SO_KEEPALIVE maintains connections during intermittent connectivity
- Optimized timeouts for better battery usage

## Security Considerations

1. Always change the default password in a production environment
2. Consider using firewall rules to restrict access
3. For production, use proper SSL certificates rather than self-signed ones
4. Review and modify logging settings for privacy and compliance
5. Monitor logs for unusual activities

## Troubleshooting

If services fail to start:
1. Check if ports are already in use
2. Verify permissions (some services may require root access)
3. Check log files in `/logs/` directory
4. Ensure Tor is running if using the Tor integration
