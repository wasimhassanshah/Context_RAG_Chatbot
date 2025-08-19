import os
import phoenix as px
import time

os.environ["PHOENIX_PORT"] = "6006"
print("🔥 Starting Phoenix on http://localhost:6006")
session = px.launch_app()
print("✅ Phoenix running - keep this terminal open")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Phoenix stopped")