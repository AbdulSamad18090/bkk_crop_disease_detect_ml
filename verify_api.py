import requests
import os
import sys
import time
import subprocess
import signal

def test_api():
    # Start API
    print("Starting API...")
    # Using Popen to start it in background
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to start
    time.sleep(10)
    
    try:
        # Check health
        try:
            resp = requests.get("http://127.0.0.1:8000/")
            print(f"Health check: {resp.status_code} - {resp.json()}")
            if resp.status_code != 200:
                print("API health check failed")
                return
        except Exception as e:
            print(f"Failed to connect to API: {e}")
            return

        # Find a test image
        test_dir = "dataset/test"
        image_path = None
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    break
            if image_path:
                break
        
        if not image_path:
            print("No test image found in dataset/test")
            return

        print(f"Testing with image: {image_path}")
        
        # Test predict
        with open(image_path, "rb") as f:
            files = {"file": f}
            resp = requests.post("http://127.0.0.1:8000/predict", files=files)
            
        print(f"Prediction status: {resp.status_code}")
        print(f"Prediction result: {resp.json()}")
        
        if resp.status_code == 200 and "class" in resp.json():
            print("SUCCESS: API returned a prediction.")
        else:
            print("FAILURE: API did not return a valid prediction.")

    finally:
        print("Stopping API...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()

if __name__ == "__main__":
    test_api()
