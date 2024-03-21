#!/bin/bash

# Checking Python version (assuming Python 3.9 is required)
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=9

echo "Verifying Python version before executing the chatbot..."

# Function to compare Python versions
version_greater_equal() {
    # Using sort -V to compare versions and checking if desired version comes last or is equal
    [[ "$1" == $(echo -e "$1\n$2" | sort -V | tail -n1) ]]
}

# Extracting the current Python version
CURRENT_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
CURRENT_PYTHON_MAJOR=$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f1)
CURRENT_PYTHON_MINOR=$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f2)

# Checking if the current Python version meets the requirement
if ! version_greater_equal "$CURRENT_PYTHON_MAJOR.$CURRENT_PYTHON_MINOR" "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"; then
    echo "Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR or higher is required."
    echo "You have Python $CURRENT_PYTHON_VERSION."
    exit 1
else
    echo "Correct Python version detected: Python $CURRENT_PYTHON_VERSION"
fi

# Optional: Navigating to the script directory if stored in other than the current working directory
#cd /path/to/your/script

# Optional: Activating virtual environment if you have one
#source /path/to/venv/bin/activate

# Installing required dependencies
echo "Installing required Python packages from requirements.txt..."
pip install -r requirements.txt

# Running the Python script
echo "Starting the chatbot..."
# Update the script name to your current chatbot version script
panel serve chatbot_v7.py --autoreload --show
