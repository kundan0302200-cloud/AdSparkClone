import os
import sys

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import app

if __name__ == "__main__":
    print("Starting AdSpark Flask Web Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
