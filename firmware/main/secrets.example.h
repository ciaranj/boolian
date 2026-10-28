// Copy to secrets.h (gitignored) and fill in.
#pragma once

#define WIFI_SSID "your-network"
#define WIFI_PASSWORD "your-password"

// The spectre container's /mic endpoint. Add ?token=... if SPECTRE_AUTH_TOKEN is set.
#define SERVER_URI "ws://192.168.1.50:8765/mic"

// Shown in the operator panel.
#define DEVICE_NAME "porch"
