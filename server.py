#!/usr/bin/env python3
"""
Simple HTTP Server for Arabcap Dashboard
Serves the dashboard on http://localhost:8000
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Configuration
PORT = 8000
DASHBOARD_DIR = "dashboard"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve from dashboard directory"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    # Check if dashboard directory exists
    if not os.path.exists(DASHBOARD_DIR):
        print(f"❌ Error: '{DASHBOARD_DIR}' directory not found!")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("=" * 60)
    print("🚀 Arabcap Dashboard Server")
    print("=" * 60)
    print(f"📂 Serving from: {os.path.join(os.getcwd(), DASHBOARD_DIR)}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print(f"🌐 Network access at: http://0.0.0.0:{PORT}")
    print()
    print("✨ Dashboard Features:")
    print("   • Material inventory monitoring")
    print("   • Automatic MIN/MAX alerts")
    print("   • Scrap prediction analytics")
    print("   • Order recommendations")
    print()
    print("⚡ Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Create server
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"✅ Server started successfully!")
            print(f"🔗 Open in browser: http://localhost:{PORT}")
            print()
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"\n❌ Error: Port {PORT} is already in use!")
            print(f"   Try closing other applications or change the PORT in this script")
        else:
            print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
