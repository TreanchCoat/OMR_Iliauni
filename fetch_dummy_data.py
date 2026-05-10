import os
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:5000/api/v1"
OUTPUT_DIR = "dummy_output"
TOKEN = "supersecrettoken"

def main():
    # 1. Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Output directory set to: {OUTPUT_DIR}")

    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }

    # 2. Trigger the process endpoint
    # The dummy API requires a non-empty payload to simulate an upload.
    print(f"[*] Calling POST {API_URL}/process to initialize the dummy job...")
    try:
        process_response = requests.post(f"{API_URL}/process", data=b"dummy_bytes", headers=headers)
        process_response.raise_for_status()
        print("    -> Process initialized successfully.")
    except requests.exceptions.RequestException as e:
        print(f"[!] Error connecting to API: {e}")
        print("    Make sure the dummy.py server is running and the token is correct!")
        return

    # 3. Fetch and save the rectified image
    print(f"[*] Fetching rectified image...")
    rectified_res = requests.get(f"{API_URL}/rectified", headers=headers)
    if rectified_res.status_code == 200:
        img_path = os.path.join(OUTPUT_DIR, "rectified.png")
        with open(img_path, "wb") as f:
            f.write(rectified_res.content)
        print(f"    -> Saved to {img_path}")
    else:
        print(f"    -> Failed to fetch rectified image: {rectified_res.status_code} - {rectified_res.text}")

    # 4. Fetch and save the XML
    print(f"[*] Fetching MusicXML...")
    xml_res = requests.get(f"{API_URL}/xml", headers=headers)
    if xml_res.status_code == 200:
        xml_path = os.path.join(OUTPUT_DIR, "score.xml")
        with open(xml_path, "wb") as f:
            f.write(xml_res.content)
        print(f"    -> Saved to {xml_path}")
    else:
        print(f"    -> Failed to fetch XML: {xml_res.status_code} - {xml_res.text}")

    # 5. Fetch and save the detections JSON
    print(f"[*] Fetching detections...")
    det_res = requests.get(f"{API_URL}/detections", headers=headers)
    if det_res.status_code == 200:
        det_path = os.path.join(OUTPUT_DIR, "detections.json")
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(det_res.json(), f, indent=4, ensure_ascii=False)
        print(f"    -> Saved to {det_path}")
    else:
        print(f"    -> Failed to fetch detections: {det_res.status_code} - {det_res.text}")

    print("\n[*] All done!")

if __name__ == "__main__":
    main()
