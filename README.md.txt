# Safe Siri Prototype

A dual-key authorization system using local AI (Ollama) for code generation and security auditing.

## Requirements
- Python 3.6+
- Ollama installed locally with mistral and codellama models
- curl

## Setup
# Go To Project
cd /d E:\AyraRachelFriday-project

# Install Python dependencies
pip install -r requirements.txt

# Download AI models
ollama pull mistral
ollama pull codellama:7b

# Run Arya
python siri_core.py


**Now users can just double-click `start_arya.bat` (Windows) or run `./start_arya.sh` (Linux) and everything will set up automatically!** 🚀

# Arya Rachel Friday - One-Click Startup

## 🪟 Windows Users:
**Double-click:** `start_arya.bat`

Or run in command prompt:
```cmd
python start_arya.py

Make executable and run:
chmod +x start_arya.sh
./start_arya.sh