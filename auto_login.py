from fyers_apiv3 import fyersModel
import webbrowser
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse

# ================= USER CREDENTIALS =================
CLIENT_ID = "paste your client id"
SECRET_KEY = "paste your secret key"
REDIRECT_URI = "http://127.0.0.1:8080/"
# ====================================================

auth_code_result = None

class AuthHandler(BaseHTTPRequestHandler):
    """Local HTTP handler to capture FYERS redirect with auth_code."""
    
    def do_GET(self):
        global auth_code_result
        parsed = urlparse.urlparse(self.path)
        params = urlparse.parse_qs(parsed.query)
        
        if "auth_code" in params:
            auth_code_result = params["auth_code"][0]
            
            # Show success message in browser
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
            <body style="display:flex;justify-content:center;align-items:center;height:100vh;font-family:Arial;background:#0d1117;color:#58a6ff;">
                <div style="text-align:center;">
                    <h1>✅ Auth Code Captured!</h1>
                    <p>Generating access token... Check your terminal.</p>
                    <p>You can close this tab now.</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif "s" in params:
            # FYERS sometimes sends auth_code as 's' parameter or in different format
            # Try to extract from the full URL
            auth_code_result = params["s"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Processing... Check terminal.</h1></body></html>")
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
            <body style="display:flex;justify-content:center;align-items:center;height:100vh;font-family:Arial;background:#0d1117;color:#f85149;">
                <div style="text-align:center;">
                    <h1>❌ Auth Code Not Found</h1>
                    <p>URL parameters: """ + str(params) + """</p>
                    <p>Please try again.</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logs."""
        pass


def update_config(access_token):
    """Update config.json in the repository root (same dir as this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    with open(config_path, "w") as f:
        json.dump({"access_token": access_token}, f, indent=2)
    
    print(f"  ✅ Updated: {config_path}")
    return [config_path]


def main():
    print("=" * 55)
    print("  🔐 GTF AUTO LOGIN — FYERS Token Generator")
    print("=" * 55)
    
    # Step 1: Generate auth URL
    print("\n📌 Step 1: Generating auth URL...")
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    auth_url = session.generate_authcode()
    print(f"  Auth URL: {auth_url}")
    
    # Step 2: Start local server
    print("\n📌 Step 2: Starting local server on port 8080...")
    server = HTTPServer(("127.0.0.1", 8080), AuthHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()
    print("  ✅ Server ready — waiting for redirect...")
    
    # Step 3: Open browser
    print("\n📌 Step 3: Opening browser for login...")
    webbrowser.open(auth_url)
    print("  👉 Please login in browser. Waiting for auth code...")
    
    # Step 4: Wait for auth code
    server_thread.join(timeout=300)  # 5 minute timeout
    server.server_close()
    
    if not auth_code_result:
        print("\n❌ ERROR: Auth code not received! (Timeout or login failed)")
        print("  Please try again.")
        return
    
    print(f"\n📌 Step 4: Auth code captured! ✅")
    print(f"  Auth Code: {auth_code_result[:20]}...")
    
    # Step 5: Generate access token
    print("\n📌 Step 5: Generating access token...")
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    session.set_token(auth_code_result)
    response = session.generate_token()
    
    if "access_token" not in response or not response["access_token"]:
        print(f"\n❌ ERROR: Token generation failed!")
        print(f"  Response: {response}")
        return
    
    access_token = response["access_token"]
    print(f"  ✅ Access token generated!")
    print(f"  Token: {access_token[:30]}...")
    
    # Step 6: Update config files
    print("\n📌 Step 6: Updating config.json files...")
    updated = update_config(access_token)
    
    print("\n" + "=" * 55)
    print("  🎉 ALL DONE! Token generated & saved!")
    print(f"  📁 Updated {len(updated)} config file(s)")
    print("  🚀 You can now run any scanner!")
    print("=" * 55)


if __name__ == "__main__":
    main()
