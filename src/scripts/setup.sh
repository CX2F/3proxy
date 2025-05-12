
#!/bin/bash
# Path: src/scripts/setup.sh

# Create required directories
mkdir -p /logs /certs /plugins /var/run/3proxy /etc/3proxy

# Generate self-signed certificate for HTTPS
openssl req -x509 -newkey rsa:4096 -keyout /certs/server.key -out /certs/server.crt -days 365 -nodes -subj "/CN=mobileproxy"

# Create admin user with secure password
echo "admin:CL:mobileproxy2024" > /etc/3proxy/passwd

# Mobile optimization settings
# Set TCP parameters for better mobile network performance
sysctl -w net.ipv4.tcp_fastopen=3 2>/dev/null || echo "TCP fastopen not available"
sysctl -w net.ipv4.tcp_slow_start_after_idle=0 2>/dev/null || echo "TCP slow start config not available"
sysctl -w net.ipv4.tcp_keepalive_time=600 2>/dev/null || echo "TCP keepalive config not available"

# Create complete mobile-optimized configuration
cat > /src/config/3proxy.cfg << EOF
# Mobile-optimized 3proxy configuration
# Path: src/config/3proxy.cfg

# DNS configuration for mobile networks
nscache 65536
nserver 1.1.1.1
nserver 8.8.8.8

# Logging configuration
log /logs/3proxy-%y%m%d.log D
logformat "L%C - %U [%t] \"%T\" %E %I %O %N/%R:%n"

# Mobile-optimized TCP parameters
# Lower timeouts and buffer sizes for mobile connections
timeout 30
maxconn 1000
daemon
stacksize 8192

# Authentication
users admin:CL:mobileproxy2024

# Mobile-friendly services (lower bandwidth, optimize for latency)
# SOCKS proxy optimized for mobile
auth strong
allow admin
socks -p5000 -i0.0.0.0 -osTCP_NODELAY

# HTTP(S) Proxy with compression
proxy -p8080 -i0.0.0.0 -n -a -osTCP_NODELAY -olSO_REUSEADDR

# Data-saving DNS proxy 
dns -p5353 -i0.0.0.0 -T

# TCP port mapping service
tcppm -i0.0.0.0 -p2222 127.0.0.1 22

# UDP port mapping service for mobile VoIP
udppm -i0.0.0.0 -p5400 127.0.0.1 5400
EOF

# Start 3proxy with the mobile-optimized configuration
echo "Starting 3proxy with mobile optimization..."
3proxy /src/config/3proxy.cfg
