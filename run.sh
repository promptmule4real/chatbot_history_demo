#!/bin/bash

# Description:
# This script verifies the Python version, installs required dependencies, and runs the chatbot application.

# Required Python version
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=9

# Function to compare Python versions
version_greater_equal() {
    # Using sort -V to compare versions and checking if desired version comes last or is equal
    [[ "$1" == $(echo -e "$1\n$2" | sort -V | tail -n1) ]]
}

# Verify Python version
echo "Verifying Python version before executing the chatbot..."
CURRENT_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
CURRENT_PYTHON_MAJOR=$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f1)
CURRENT_PYTHON_MINOR=$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f2)

if ! version_greater_equal "$CURRENT_PYTHON_MAJOR.$CURRENT_PYTHON_MINOR" "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"; then
    echo "Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR or higher is required."
    echo "You have Python $CURRENT_PYTHON_VERSION."
    exit 1
else
    echo "Correct Python version detected: Python $CURRENT_PYTHON_VERSION"
fi

# Navigate to the script directory (if necessary)
# cd /path/to/your/script

# Activate virtual environment (if applicable)
# source /path/to/venv/bin/activate

# Install required dependencies
echo "Installing required Python packages from requirements.txt..."
pip install -r requirements.txt

# Run the chatbot application
echo "Starting the chatbot..."
panel serve chatbot_v7.py --autoreload --show