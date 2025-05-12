#!/bin/bash
# Path: src/scripts/setup.sh

# Create required directories
mkdir -p /logs /certs /plugins /var/run/3proxy /etc/3proxy

# Generate self-signed certificate for HTTPS
openssl req -x509 -newkey rsa:4096 -keyout /certs/server.key -out /certs/server.crt -days 365 -nodes -subj "/CN=mobileproxy"

# Create admin user
echo "admin:CL:mobileproxy" > /etc/3proxy/passwd

# Install Tor if not present (optional)
if ! command -v tor &> /dev/null; then
    echo "Tor not found. Installing Tor..."
    apt-get update && apt-get install -y tor
    # Start tor service
    service tor start || echo "Could not start Tor service, continuing anyway"
fi

# Optimize kernel parameters for mobile connections
sysctl -w net.ipv4.tcp_fastopen=3 || echo "Could not set TCP fastopen, continuing anyway"
sysctl -w net.core.somaxconn=1024 || echo "Could not increase connection backlog, continuing anyway"

# Configure firewall (if available)
if command -v iptables &> /dev/null; then
    echo "Configuring iptables..."
    iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080 || echo "iptables configuration failed, continuing anyway"
fi

# Start 3proxy with the mobile-optimized configuration
echo "Starting 3proxy with mobile optimization..."
3proxy /src/config/3proxy.cfg