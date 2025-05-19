
#!/bin/bash
# cleanup.sh - Script to remove temporary and unnecessary files

echo "Cleaning up project directory..."

# Remove temporary files
find . -name "*.o" -type f -delete
find . -name "*.blob" -type f -delete
find . -name "*.tmp" -type f -delete
find . -name "*.bak" -type f -delete

# Remove cache directories
if [ -d ".ccls-cache" ]; then
  echo "Removing .ccls-cache..."
  rm -rf .ccls-cache
fi

if [ -d ".config" ]; then
  echo "Removing .config..."
  rm -rf .config
fi

# Clean empty log files
if [ -d "logs" ]; then
  find logs -type f -size 0 -delete
fi

echo "Cleanup complete!"
