import requests
import msal
import os
from pathlib import Path

# -------------------------
# AZURE CONFIG
# -------------------------

TENANT_ID = "YOUR_TENANT_ID"
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

SHAREPOINT_SITE_NAME = "site-name"
SHAREPOINT_DOCUMENT_LIBRARY = "Shared Documents"

TEMP_DOWNLOAD_DIR = "temp_videos"
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

def get_access_token():
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=authority,
        client_credential=CLIENT_SECRET
    )

    token_response = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )

    return token_response["access_token"]

def get_site_id(token):
    url = f"https://graph.microsoft.com/v1.0/sites/company.sharepoint.com:/sites/{SHAREPOINT_SITE_NAME}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    return response.json()["id"]

def list_video_files(token, site_id):
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    files = response.json()["value"]

    video_files = [
        f for f in files
        if f["name"].lower().endswith(".mp4")
    ]

    return video_files

def download_video(token, file_id, filename):
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, stream=True)

    local_path = Path(TEMP_DOWNLOAD_DIR) / filename

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    return str(local_path)

token = get_access_token()
site_id = get_site_id(token)
video_files = list_video_files(token, site_id)

for file in video_files:

    video_name = file["name"]
    file_id = file["id"]
    video_id = Path(video_name).stem

    print(f"Processing {video_name}")

    # Download temporarily
    local_video_path = download_video(token, file_id, video_name)

    # Transcribe
    segments = transcribe_video(local_video_path)

    # Delete temp file
    os.remove(local_video_path)

    chunks = chunk_transcript(segments, CHUNK_SIZE_SEC)

    for idx, chunk in enumerate(chunks):
        embedding = embed(chunk["text"])
        all_embeddings.append(embedding)

        all_metadata.append({
            "video_id": video_id,
            "video_url": file["webUrl"],  # 🔥 direct SharePoint URL
            "chunk_id": idx,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        })
