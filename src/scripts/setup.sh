
#!/bin/bash
# Path: src/scripts/setup.sh

# Create required directories
mkdir -p /logs /certs /plugins /var/run/3proxy /etc/3proxy

# Generate self-signed certificate for HTTPS
openssl req -x509 -newkey rsa:4096 -keyout /certs/server.key -out /certs/server.crt -days 365 -nodes -subj "/CN=localhost"

# Create admin user
echo "admin:CL:password123" > /etc/3proxy/passwd

# Install Tor if not present (optional)
if ! command -v tor &> /dev/null; then
    echo "Tor not found. Installing Tor..."
    apt-get update && apt-get install -y tor
    # Start tor service
    service tor start
fi

# Create custom TLD configuration (for internal testing only)
mkdir -p /etc/3proxy/dns
cat > /etc/3proxy/dns/mytld.zone << EOF
; Custom TLD zone file
\$TTL 86400
@ IN SOA ns1.mytld. admin.mytld. (
    2023101301 ; serial
    3600       ; refresh
    1800       ; retry
    604800     ; expire
    86400      ; minimum TTL
)
@       IN NS ns1.mytld.
ns1     IN A  127.0.0.1
www     IN A  127.0.0.1
test    IN A  127.0.0.1
EOF

# Configure iptables for port forwarding (requires root)
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null || echo "Cannot enable IP forwarding (not running as root)"

# Start 3proxy with the configuration
echo "Starting 3proxy..."
3proxy /src/config/3proxy.cfg
