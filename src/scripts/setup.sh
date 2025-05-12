
#!/bin/bash

# Create required directories
mkdir -p /logs /certs /plugins

# Generate self-signed certificate for HTTPS
openssl req -x509 -newkey rsa:4096 -keyout /certs/server.key -out /certs/server.crt -days 365 -nodes -subj "/CN=localhost"

# Create admin user
echo "admin:CL:password123" > /etc/3proxy/passwd

# Start 3proxy
3proxy /src/config/3proxy.cfg
