import os

import requests
from google.cloud import storage


DRIVE_FILE_URL = 'https://storage.googleapis.com/cs-6375-cloth/computer%2Bhardware/'
BUCKET_NAME = 'cs-6375-cloth'
PREFIX = 'computer+hardware/'
LOCAL_FOLDER = '.'



def stdout_output(*args):
    with open('log.txt', 'a') as log_file:  # Open the log file in append mode
        if args:
            print(*args)
            log_file.write(' '.join(map(str, args)) + '\n')  # Convert args to strings and write to the log file
        else:
            print()
            log_file.write('\n')


def download_folder():
    """
    Downloads a "folder" from GCS, which means all objects with a common prefix.
    """

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=PREFIX)

    for blob in blobs:
        filename = blob.name.replace('/', '_')  # Convert the blob name into a suitable local filename
        local_file_path = os.path.join(LOCAL_FOLDER, filename)

        # Check if the file already exists locally
        if not os.path.exists(local_file_path):
            blob.download_to_filename(local_file_path)

