#!/bin/bash
# Run with: sudo bash install_service.sh

echo "Installing KEN Cloud Server as systemd service..."

sudo cp ken-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ken-server
sudo systemctl start ken-server

echo "Service installed and started."
echo "Check status: sudo systemctl status ken-server"
echo "View logs:    sudo journalctl -u ken-server -f"
