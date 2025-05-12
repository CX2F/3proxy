
# Path: src/Makefile
# Makefile for mobile proxy

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -pthread
LDFLAGS = -pthread

# Directories
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = ../bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/mobile_proxy

# Default target
all: directories $(TARGET)

# Create directories
directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Install
install: all
	cp $(TARGET) /usr/local/bin/

.PHONY: all directories clean install
# Path: src/Makefile
# Makefile for mobile proxy utility

CXX = g++
CXXFLAGS = -std=c++11 -Wall -pthread -O2
LDFLAGS = -pthread

# Output directories
BINDIR = ../bin
OBJDIR = ../obj

# Make sure directories exist
$(shell mkdir -p $(BINDIR) $(OBJDIR))

# Targets
all: mobile_proxy proxy_server

mobile_proxy: $(OBJDIR)/mobile_proxy.o
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/mobile_proxy $(OBJDIR)/mobile_proxy.o $(LDFLAGS)

proxy_server: $(OBJDIR)/server.o
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/proxy_server $(OBJDIR)/server.o $(LDFLAGS)

$(OBJDIR)/mobile_proxy.o: mobile_proxy.cpp
	$(CXX) $(CXXFLAGS) -c mobile_proxy.cpp -o $(OBJDIR)/mobile_proxy.o

$(OBJDIR)/server.o: server.cpp
	$(CXX) $(CXXFLAGS) -c server.cpp -o $(OBJDIR)/server.o

clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/mobile_proxy $(BINDIR)/proxy_server

.PHONY: all clean
