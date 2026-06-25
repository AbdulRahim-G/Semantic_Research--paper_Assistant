import os
import sys

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath("backend"))
os.environ["PYTHONPATH"] = os.path.abspath("backend")
os.environ["PYTHONUNBUFFERED"] = "1"

# pyrefly: ignore [missing-import]
import uvicorn

if __name__ == "__main__":
    print("Starting backend server via uvicorn...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
