
// Path: src/server.cpp
// Server implementation for mobile proxy client connections

#include <iostream>
#include <string>
#include <thread>
#include <map>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

// Constants
const int SERVER_PORT = 9090;
const int MAX_CLIENTS = 100;
const int BUFFER_SIZE = 4096;

// Server state
std::atomic<bool> running(false);
std::map<int, std::thread> client_threads;
std::mutex clients_mutex;
int server_socket = -1;

// Forward declaration
void handleClient(int client_socket, const std::string& client_ip);

// Socket optimization
void optimizeSocket(int socket) {
    // Enable TCP_NODELAY (disable Nagle's algorithm)
    int flag = 1;
    setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    // Enable keepalive
    setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    
    // Set receive buffer size
    int rcvbuf = 65536; // 64KB
    setsockopt(socket, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    // Set send buffer size
    int sndbuf = 65536; // 64KB
    setsockopt(socket, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
}

// Signal handler
void signalHandler(int signal) {
    std::cout << "Received signal " << signal << std::endl;
    running = false;
    
    // Close server socket to unblock accept()
    if (server_socket >= 0) {
        close(server_socket);
        server_socket = -1;
    }
}

// Start the server
bool startServer() {
    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Failed to create server socket" << std::endl;
        return false;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Failed to set SO_REUSEADDR" << std::endl;
        close(server_socket);
        return false;
    }
    
    // Apply optimizations
    optimizeSocket(server_socket);
    
    // Bind socket
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind server socket to port " << SERVER_PORT << std::endl;
        close(server_socket);
        return false;
    }
    
    // Listen for connections
    if (listen(server_socket, 10) < 0) {
        std::cerr << "Failed to listen on server socket" << std::endl;
        close(server_socket);
        return false;
    }
    
    running = true;
    std::cout << "Server started on port " << SERVER_PORT << std::endl;
    
    return true;
}

// Accept client connections
void acceptClients() {
    while (running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            if (running) {
                std::cerr << "Failed to accept client connection" << std::endl;
            }
            continue;
        }
        
        // Apply optimizations to client socket
        optimizeSocket(client_socket);
        
        // Get client IP
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        
        std::cout << "Client connected from " << client_ip << std::endl;
        
        // Start client thread
        std::lock_guard<std::mutex> lock(clients_mutex);
        client_threads[client_socket] = std::thread(handleClient, client_socket, std::string(client_ip));
    }
}

// Client request handler
void handleCommand(const std::string& command, std::string& response) {
    if (command == "STATUS") {
        response = "ONLINE";
    } else if (command == "PORTS") {
        response = "SOCKS:5000,HTTP:8080,DNS:5353,SSH:2222,VoIP:5400,Server:9090";
    } else if (command == "PING") {
        response = "PONG";
    } else if (command == "STATS") {
        // Get connection stats
        std::lock_guard<std::mutex> lock(clients_mutex);
        response = "Active clients: " + std::to_string(client_threads.size());
    } else if (command.substr(0, 8) == "CONNECT:") {
        // Handle proxy connection request
        std::string target = command.substr(8);
        response = "CONNECTING TO " + target;
        // Actual connection handling would go here
    } else {
        response = "UNKNOWN COMMAND";
    }
}

// Handle client connection
void handleClient(int client_socket, const std::string& client_ip) {
    try {
        char buffer[BUFFER_SIZE];
        
        // Simple protocol: client sends commands, server responds
        while (running) {
            // Clear buffer
            memset(buffer, 0, BUFFER_SIZE);
            
            // Receive data
            int bytes_read = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);
            if (bytes_read <= 0) {
                // Connection closed or error
                break;
            }
            
            // Process command
            std::string command(buffer, bytes_read);
            std::string response;
            
            handleCommand(command, response);
            
            // Send response
            send(client_socket, response.c_str(), response.length(), 0);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling client " << client_ip << ": " << e.what() << std::endl;
    }
    
    // Remove from client threads
    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        client_threads.erase(client_socket);
    }
    
    // Close socket
    close(client_socket);
    std::cout << "Client disconnected: " << client_ip << std::endl;
}

// Stop the server
void stopServer() {
    if (running) {
        running = false;
        
        // Close server socket
        if (server_socket >= 0) {
            close(server_socket);
            server_socket = -1;
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
        
        std::cout << "Server stopped" << std::endl;
    }
}

// Main function
int main() {
    // Set up signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "Starting mobile proxy server..." << std::endl;
    
    if (!startServer()) {
        return 1;
    }
    
    // Accept clients in the main thread
    acceptClients();
    
    // Clean up
    stopServer();
    
    return 0;
}
