
// Path: src/mobile_proxy.cpp
// Mobile-optimized proxy utility with C++

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <chrono>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <sstream>
#include <ctime>

// Network headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

// Enhanced logging class
class Logger {
private:
    std::string log_file;
    std::mutex log_mutex;
    std::ofstream log_stream;
    bool console_output;
    
public:
    enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    Logger(const std::string& file = "mobile_proxy.log", bool console = true) 
        : log_file(file), console_output(console) {
        try {
            log_stream.open(log_file, std::ios::app);
            if (!log_stream) {
                std::cerr << "Failed to open log file: " << log_file << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error initializing logger: " << e.what() << std::endl;
        }
    }
    
    ~Logger() {
        if (log_stream.is_open()) {
            log_stream.close();
        }
    }
    
    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        
        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&time, &tm_buf);
        
        char time_str[26];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
        
        // Convert log level to string
        std::string level_str;
        switch (level) {
            case DEBUG: level_str = "DEBUG"; break;
            case INFO: level_str = "INFO"; break;
            case WARNING: level_str = "WARNING"; break;
            case ERROR: level_str = "ERROR"; break;
            case CRITICAL: level_str = "CRITICAL"; break;
        }
        
        // Format log message
        std::ostringstream log_msg;
        log_msg << "[" << time_str << "] [" << level_str << "] " << message;
        
        // Write to file
        if (log_stream.is_open()) {
            log_stream << log_msg.str() << std::endl;
            log_stream.flush();
        }
        
        // Write to console if enabled
        if (console_output) {
            if (level == ERROR || level == CRITICAL) {
                std::cerr << log_msg.str() << std::endl;
            } else {
                std::cout << log_msg.str() << std::endl;
            }
        }
    }
    
    void debug(const std::string& message) { log(DEBUG, message); }
    void info(const std::string& message) { log(INFO, message); }
    void warning(const std::string& message) { log(WARNING, message); }
    void error(const std::string& message) { log(ERROR, message); }
    void critical(const std::string& message) { log(CRITICAL, message); }
};

// Global logger
Logger logger("./logs/mobile_proxy.log");

// Service monitoring
class ServiceMonitor {
private:
    std::mutex mtx;
    struct ServiceInfo {
        bool running;
        int port;
        std::string type;
        time_t start_time;
        int connections;
    };
    std::map<std::string, ServiceInfo> services;
    
public:
    void registerService(const std::string& name, int port, const std::string& type) {
        std::lock_guard<std::mutex> lock(mtx);
        services[name] = {true, port, type, time(nullptr), 0};
        logger.info("Service registered: " + name + " on port " + std::to_string(port));
    }
    
    void updateService(const std::string& name, bool running) {
        std::lock_guard<std::mutex> lock(mtx);
        if (services.find(name) != services.end()) {
            services[name].running = running;
            logger.info("Service " + name + " status updated: " + (running ? "running" : "stopped"));
        }
    }
    
    void incrementConnections(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx);
        if (services.find(name) != services.end()) {
            services[name].connections++;
        }
    }
    
    void decrementConnections(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx);
        if (services.find(name) != services.end() && services[name].connections > 0) {
            services[name].connections--;
        }
    }
    
    void printStatus() {
        std::lock_guard<std::mutex> lock(mtx);
        logger.info("=== Service Status ===");
        for (const auto& pair : services) {
            const auto& name = pair.first;
            const auto& info = pair.second;
            
            time_t uptime = time(nullptr) - info.start_time;
            
            std::ostringstream ss;
            ss << name << " (" << info.type << ") - Port: " << info.port 
               << " - Status: " << (info.running ? "Running" : "Stopped")
               << " - Uptime: " << uptime << "s"
               << " - Active connections: " << info.connections;
            
            logger.info(ss.str());
        }
        logger.info("=====================");
    }
};

// Config class
class MobileProxyConfig {
public:
    std::string config_file;
    std::map<std::string, std::string> settings;
    
    MobileProxyConfig(const std::string& file = "./src/config/3proxy.cfg") 
        : config_file(file) {
        loadConfig();
    }
    
    void loadConfig() {
        try {
            std::ifstream file(config_file);
            if (!file) {
                logger.error("Cannot open config file " + config_file);
                // Create a default config
                createDefaultConfig();
                return;
            }
            
            std::string line;
            while (std::getline(file, line)) {
                // Skip comments and empty lines
                if (line.empty() || line[0] == '#') continue;
                
                // Parse key=value pairs
                size_t pos = line.find(' ');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);
                    settings[key] = value;
                }
            }
            logger.info("Config loaded: " + std::to_string(settings.size()) + " settings");
        } catch (const std::exception& e) {
            logger.error("Error loading config: " + std::string(e.what()));
        }
    }
    
    void createDefaultConfig() {
        logger.info("Creating default configuration...");
        try {
            // Create directories
            system("mkdir -p ./src/config");
            
            std::ofstream file(config_file);
            if (!file) {
                logger.error("Cannot create config file " + config_file);
                return;
            }
            
            file << "# Mobile-optimized 3proxy configuration\n"
                 << "# Path: src/config/3proxy.cfg\n\n"
                 << "# DNS configuration for mobile networks\n"
                 << "nscache 65536\n"
                 << "nserver 1.1.1.1\n"
                 << "nserver 8.8.8.8\n\n"
                 << "# Logging configuration\n"
                 << "log ./logs/3proxy-%y%m%d.log D\n"
                 << "logformat \"L%C - %U [%t] \\\"%T\\\" %E %I %O %N/%R:%n\"\n\n"
                 << "# Mobile-optimized TCP parameters\n"
                 << "timeout 30\n"
                 << "maxconn 1000\n"
                 << "daemon\n"
                 << "stacksize 8192\n\n"
                 << "# Authentication\n"
                 << "users admin:CL:mobileproxy2024\n\n"
                 << "# Mobile-friendly services\n"
                 << "auth strong\n"
                 << "allow admin\n"
                 << "socks -p5000 -i0.0.0.0 -osTCP_NODELAY\n\n"
                 << "# HTTP(S) Proxy with compression\n"
                 << "proxy -p8080 -i0.0.0.0 -n -a -osTCP_NODELAY -olSO_REUSEADDR\n\n"
                 << "# Data-saving DNS proxy\n" 
                 << "dns -p5353 -i0.0.0.0 -T\n\n"
                 << "# TCP port mapping service\n"
                 << "tcppm -i0.0.0.0 -p2222 127.0.0.1 22\n\n"
                 << "# UDP port mapping service for mobile VoIP\n"
                 << "udppm -i0.0.0.0 -p5400 127.0.0.1 5400\n";
            
            file.close();
            logger.info("Default configuration created at " + config_file);
            
            // Now load the config we just created
            loadConfig();
        } catch (const std::exception& e) {
            logger.error("Error creating default config: " + std::string(e.what()));
        }
    }
};

// Connection monitoring
class ConnectionMonitor {
private:
    std::mutex mtx;
    std::map<std::string, int> connections;
    int max_connections;
    
public:
    ConnectionMonitor(int max = 100) : max_connections(max) {}
    
    bool addConnection(const std::string& ip) {
        std::lock_guard<std::mutex> lock(mtx);
        if (connections[ip] >= max_connections) {
            logger.warning("Connection limit reached for IP: " + ip);
            return false;
        }
        connections[ip]++;
        return true;
    }
    
    void removeConnection(const std::string& ip) {
        std::lock_guard<std::mutex> lock(mtx);
        if (connections[ip] > 0) {
            connections[ip]--;
        }
    }
    
    void printStats() {
        std::lock_guard<std::mutex> lock(mtx);
        logger.info("=== Connection Statistics ===");
        for (const auto& pair : connections) {
            logger.info(pair.first + ": " + std::to_string(pair.second) + " connections");
        }
        logger.info("===========================");
    }
};

// Mobile optimization utility
class MobileOptimizer {
public:
    static void optimizeSocket(int socket) {
        try {
            // Enable TCP_NODELAY (disable Nagle's algorithm)
            int flag = 1;
            if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) < 0) {
                logger.warning("Could not set TCP_NODELAY");
            }
            
            // Enable keepalive
            if (setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag)) < 0) {
                logger.warning("Could not set SO_KEEPALIVE");
            }
            
            // Set receive buffer size for mobile optimization
            int rcvbuf = 65536; // 64KB
            if (setsockopt(socket, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
                logger.warning("Could not set SO_RCVBUF");
            }
            
            // Set send buffer size for mobile optimization
            int sndbuf = 65536; // 64KB
            if (setsockopt(socket, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
                logger.warning("Could not set SO_SNDBUF");
            }
        } catch (const std::exception& e) {
            logger.error("Error optimizing socket: " + std::string(e.what()));
        }
    }
    
    static bool isLowBandwidthMode() {
        // Check system status or config for low bandwidth mode
        return false; // Default to normal mode
    }
    
    static void adaptiveCompression(std::vector<char>& data) {
        // Implement adaptive compression based on network conditions
        if (isLowBandwidthMode() && data.size() > 4096) {
            // Simulate compression for demonstration
            data.resize(data.size() * 0.7);
            logger.debug("Compressed data from " + std::to_string(data.size()/0.7) + " to " + std::to_string(data.size()) + " bytes");
        }
    }
};

// Server implementation for client connection back
class ProxyServer {
private:
    int server_socket;
    int server_port;
    bool running;
    std::thread server_thread;
    std::map<int, std::thread> client_threads;
    std::mutex clients_mutex;
    ServiceMonitor& service_monitor;
    
public:
    ProxyServer(ServiceMonitor& monitor, int port = 6666) 
        : server_port(port), running(false), service_monitor(monitor) {
        server_socket = -1;
    }
    
    ~ProxyServer() {
        stop();
    }
    
    bool start() {
        // Create socket
        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket < 0) {
            logger.error("Failed to create server socket");
            return false;
        }
        
        // Set socket options
        int opt = 1;
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            logger.error("Failed to set SO_REUSEADDR");
            close(server_socket);
            return false;
        }
        
        // Apply mobile optimizations
        MobileOptimizer::optimizeSocket(server_socket);
        
        // Bind socket
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(server_port);
        
        if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            logger.error("Failed to bind server socket to port " + std::to_string(server_port));
            close(server_socket);
            return false;
        }
        
        // Listen for connections
        if (listen(server_socket, 10) < 0) {
            logger.error("Failed to listen on server socket");
            close(server_socket);
            return false;
        }
        
        running = true;
        
        // Register with service monitor
        service_monitor.registerService("Callback Server", server_port, "TCP");
        
        // Start server thread
        server_thread = std::thread(&ProxyServer::acceptLoop, this);
        logger.info("Proxy server started on port " + std::to_string(server_port));
        
        return true;
    }
    
    void stop() {
        if (running) {
            running = false;
            
            // Close server socket to unblock accept()
            if (server_socket >= 0) {
                close(server_socket);
                server_socket = -1;
            }
            
            // Join server thread
            if (server_thread.joinable()) {
                server_thread.join();
            }
            
            // Close all client connections
            std::lock_guard<std::mutex> lock(clients_mutex);
            for (auto& pair : client_threads) {
                if (pair.second.joinable()) {
                    pair.second.detach();
                }
                close(pair.first);
            }
            client_threads.clear();
            
            logger.info("Proxy server stopped");
        }
    }
    
private:
    void acceptLoop() {
        while (running) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket < 0) {
                if (running) {
                    logger.error("Failed to accept client connection");
                }
                continue;
            }
            
            // Apply mobile optimizations to client socket
            MobileOptimizer::optimizeSocket(client_socket);
            
            // Get client IP
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            
            logger.info("Client connected from " + std::string(client_ip));
            
            // Start client thread
            std::lock_guard<std::mutex> lock(clients_mutex);
            client_threads[client_socket] = std::thread(&ProxyServer::handleClient, this, client_socket, std::string(client_ip));
        }
    }
    
    void handleClient(int client_socket, const std::string& client_ip) {
        // Add to connection monitor
        service_monitor.incrementConnections("Callback Server");
        
        try {
            // Handle client connection
            const int buffer_size = 4096;
            char buffer[buffer_size];
            
            // Simple protocol: client sends commands, server responds
            while (running) {
                // Clear buffer
                memset(buffer, 0, buffer_size);
                
                // Receive data
                int bytes_read = recv(client_socket, buffer, buffer_size - 1, 0);
                if (bytes_read <= 0) {
                    // Connection closed or error
                    break;
                }
                
                // Process command
                std::string command(buffer, bytes_read);
                std::string response;
                
                if (command == "STATUS") {
                    response = "ONLINE";
                } else if (command == "PORTS") {
                    response = "5000,8080,5353,2222,5400";
                } else if (command == "PING") {
                    response = "PONG";
                } else {
                    response = "UNKNOWN COMMAND";
                }
                
                // Send response
                send(client_socket, response.c_str(), response.length(), 0);
            }
        } catch (const std::exception& e) {
            logger.error("Error handling client " + client_ip + ": " + e.what());
        }
        
        // Remove from client threads
        {
            std::lock_guard<std::mutex> lock(clients_mutex);
            client_threads.erase(client_socket);
        }
        
        // Close socket
        close(client_socket);
        
        // Decrement connection count
        service_monitor.decrementConnections("Callback Server");
        
        logger.info("Client disconnected: " + client_ip);
    }
};

// Mobile Proxy Controller
class MobileProxyController {
private:
    MobileProxyConfig config;
    ConnectionMonitor connection_monitor;
    ServiceMonitor service_monitor;
    ProxyServer* server;
    bool running;
    std::thread stats_thread;
    
public:
    MobileProxyController() : running(false), server(nullptr) {}
    
    ~MobileProxyController() {
        stop();
    }
    
    void start() {
        logger.info("Starting Mobile Proxy Controller...");
        running = true;
        
        // Create required directories
        system("mkdir -p ./logs ./certs ./etc/3proxy");
        
        // Generate script
        createSetupScript();
        
        // Make script executable
        system("chmod +x ./src/scripts/setup.sh");
        
        // Run setup script
        system("./src/scripts/setup.sh");
        
        // Start callback server
        server = new ProxyServer(service_monitor);
        if (!server->start()) {
            logger.error("Failed to start proxy server");
        }
        
        // Start stats monitoring thread
        stats_thread = std::thread([this]() {
            while (running) {
                std::this_thread::sleep_for(std::chrono::seconds(60));
                connection_monitor.printStats();
                service_monitor.printStatus();
            }
        });
        
        logger.info("Mobile Proxy Controller started");
    }
    
    void stop() {
        if (running) {
            logger.info("Stopping Mobile Proxy Controller...");
            running = false;
            
            // Stop 3proxy
            system("pkill 3proxy 2>/dev/null || true");
            
            // Stop server
            if (server) {
                server->stop();
                delete server;
                server = nullptr;
            }
            
            // Join stats thread
            if (stats_thread.joinable()) {
                stats_thread.join();
            }
            
            logger.info("Mobile Proxy Controller stopped");
        }
    }
    
    void reload() {
        logger.info("Reloading configuration...");
        config.loadConfig();
        
        // Signal 3proxy to reload
        system("pkill -HUP 3proxy 2>/dev/null || true");
        
        logger.info("Configuration reloaded");
    }
    
private:
    void createSetupScript() {
        try {
            // Create directory
            system("mkdir -p ./src/scripts");
            
            std::ofstream file("./src/scripts/setup.sh");
            if (!file) {
                logger.error("Cannot create setup script");
                return;
            }
            
            file << "#!/bin/bash\n"
                 << "# Path: src/scripts/setup.sh\n\n"
                 << "# Create required directories\n"
                 << "mkdir -p ./logs ./certs ./etc/3proxy\n\n"
                 << "# Generate self-signed certificate for HTTPS\n"
                 << "if [ ! -f ./certs/server.key ]; then\n"
                 << "    openssl req -x509 -newkey rsa:4096 -keyout ./certs/server.key -out ./certs/server.crt -days 365 -nodes -subj \"/CN=mobileproxy\" 2>/dev/null\n"
                 << "    echo \"Generated SSL certificates\"\n"
                 << "fi\n\n"
                 << "# Create admin user with secure password\n"
                 << "echo \"admin:CL:mobileproxy2024\" > ./etc/3proxy/passwd\n\n"
                 << "# Mobile optimization settings\n"
                 << "# Set TCP parameters for better mobile network performance\n"
                 << "sysctl -w net.ipv4.tcp_fastopen=3 2>/dev/null || echo \"TCP fastopen not available\"\n"
                 << "sysctl -w net.ipv4.tcp_slow_start_after_idle=0 2>/dev/null || echo \"TCP slow start config not available\"\n"
                 << "sysctl -w net.ipv4.tcp_keepalive_time=600 2>/dev/null || echo \"TCP keepalive config not available\"\n\n"
                 << "# Start 3proxy with the mobile-optimized configuration\n"
                 << "echo \"Starting 3proxy with mobile optimization...\"\n"
                 << "3proxy ./src/config/3proxy.cfg &\n"
                 << "echo \"3proxy started\"\n";
            
            file.close();
            logger.info("Setup script created at ./src/scripts/setup.sh");
        } catch (const std::exception& e) {
            logger.error("Error creating setup script: " + std::string(e.what()));
        }
    }
};

// Signal handler
MobileProxyController* global_controller = nullptr;

void signalHandler(int signal) {
    logger.info("Received signal " + std::to_string(signal));
    if (global_controller) {
        if (signal == SIGHUP) {
            global_controller->reload();
        } else {
            global_controller->stop();
            exit(0);
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    logger.info("Mobile-Optimized 3proxy Utility");
    logger.info("===============================");
    
    // Create directories
    system("mkdir -p ./logs");
    
    // Register signal handlers
    MobileProxyController controller;
    global_controller = &controller;
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGHUP, signalHandler);
    
    // Start controller
    controller.start();
    
    // Wait for termination
    logger.info("Controller running. Press Ctrl+C to stop.");
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
