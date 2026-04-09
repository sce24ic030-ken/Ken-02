# KEN Cloud Server — Oracle Cloud Free Tier Setup
# 100% FREE. 24/7. FOREVER. No credit card needed after signup.

## Step 1: Create Oracle Cloud Account
# https://www.oracle.com/cloud/free/
# - Sign up (free, no credit card charged)
# - Choose "Always Free" resources

## Step 2: Create ARM Instance (Always Free)
# - Go to: Compute → Instances → Create Instance
# - Shape: VM.Standard.A1.Flex (ARM, Always Free)
# - OCPU: 4 (max free)
# - Memory: 24 GB (max free)
# - Image: Ubuntu 22.04 Minimal
# - Storage: 200 GB boot volume (Always Free)
# - Add SSH key (generate or paste your public key)
# - Create

## Step 3: Open Firewall Port
# - Go to: Networking → Virtual Cloud Networks → your VCN
# - Security Lists → Default → Add Ingress Rule
# - Source CIDR: 0.0.0.0/0
# - Destination Port: 8080
# - Save

## Step 4: SSH In and Setup
# ssh ubuntu@<YOUR_INSTANCE_IP>

## Step 5: Run This Script
# Copy this entire file to the server and run:
#   bash setup.sh

#!/bin/bash
set -e

echo "=== KEN Cloud Server Setup ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git ffmpeg

# Clone repo (or upload files)
# git clone https://github.com/YOUR_USER/KenAIRobot.git
# cd KenAIRobot/ken-server

# OR just upload ken-server folder via scp:
# scp -r ken-server/ ubuntu@<IP>:~/

cd ~/ken-server

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables (optional — for AI providers)
echo '# Optional: Set your free API keys here
export GEMINI_API_KEY=""
export GROQ_API_KEY=""
export WHISPER_MODEL="base"
export PORT=8080
' > .env

echo ""
echo "=== Setup Complete ==="
echo "Start the server:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Or install as systemd service:"
echo "  sudo bash install_service.sh"
