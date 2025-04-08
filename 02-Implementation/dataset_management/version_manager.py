import os
import json
import re
import boto3
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
METADATA_BUCKET = os.getenv("METADATA_BUCKET", os.getenv("S3_BUCKET_NAME"))
METADATA_KEY = os.getenv("METADATA_KEY", "versioning/metadata.json")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def download_metadata() -> dict:
    try:
        obj = s3.get_object(Bucket=METADATA_BUCKET, Key=METADATA_KEY)
        return json.loads(obj['Body'].read())
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to download metadata: {e}")
        return {}

def upload_metadata(data: dict):
    s3.put_object(
        Bucket=METADATA_BUCKET,
        Key=METADATA_KEY,
        Body=json.dumps(data, indent=4).encode('utf-8'),
        ContentType="application/json"
    )

def get_latest_version(project: str, data: dict) -> Optional[str]:
    versions = data.get(project, [])
    return versions[-1]["version"] if versions else None

def increment_version(version: str) -> str:
    match = re.match(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("Invalid version format. Use vX.Y.Z")
    major, minor, patch = map(int, match.groups())
    return f"v{major}.{minor}.{patch + 1}"

def generate_new_version(project: str, notes: str = "") -> str:
    data = download_metadata()
    latest = get_latest_version(project, data)
    new_version = increment_version(latest) if latest else "v1.0.0"

    metadata = {
        "version": new_version,
        "timestamp": datetime.utcnow().isoformat(),
        "notes": notes
    }

    data.setdefault(project, []).append(metadata)
    upload_metadata(data)
    return new_version

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and manage build versions")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--notes", default="", help="Changelog or notes for this version")

    args = parser.parse_args()
    version = generate_new_version(args.project, args.notes)
    print(f"[✓] New version for '{args.project}': {version}")
