#!/Users/elle/.doom.d/gslides/.venv/bin/python3
"""Upload a video to YouTube using Google OAuth credentials from keychain.

Usage:
    python youtube_upload.py <video_file> [title] [description]
"""

import sys
import os

# Add the google-auth script's directory so we can import get_credentials
sys.path.insert(0, os.path.expanduser("~/scripts/claude"))
from importlib.machinery import SourceFileLoader
google_auth = SourceFileLoader("google_auth", os.path.expanduser("~/scripts/claude/google-auth")).load_module()

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload(video_path, title, description):
    creds = google_auth.get_credentials()
    if not creds:
        print("No valid credentials. Run: google-auth refresh", file=sys.stderr)
        sys.exit(1)

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": "unlisted",
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  Upload {int(status.progress() * 100)}%")

    video_id = response["id"]
    print(f"Uploaded: https://www.youtube.com/watch?v={video_id}")
    return video_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "ASCII Ecology — LLM Soup (32×32 Personality)"
    description = sys.argv[3] if len(sys.argv) > 3 else (
        "100 cells of random ASCII art evolve via Claude Haiku 4.5 prefill. "
        "Each epoch: 50 random pairs are concatenated as assistant prefill, "
        "the model continues, and the output is split back into two cells. "
        "Replicating Agüera y Arcas et al. 2024 with an LLM instead of a BFF interpreter."
    )

    upload(video_path, title, description)
