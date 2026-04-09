#!/bin/bash

echo "Setting up Echocardiogram CNN environment..."
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "To recreate, delete the venv folder first."
    echo ""
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
