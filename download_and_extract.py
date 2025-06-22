import os
import tarfile
import zipfile
import requests
from tqdm import tqdm

def download_file(url, destination_path):
    """
    Download a file from the specified URL and save it to the destination path.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination_path, 'wb') as file, tqdm(
        desc=destination_path,
        total=total_size,
        unit='B',
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)

def extract_archive(archive_path, extract_to):
    """
    Extract the archive file (tar or zip) to the specified directory.
    """
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(path=extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(path=extract_to)
    else:
        raise ValueError("Unsupported archive format. Only .tar.gz and .zip are supported.")

def download_and_extract_archive(url, download_root, filename, md5=None):
    """
    Download and extract an archive from a URL to a specified directory.
    """
    archive_path = os.path.join(download_root, filename)
    
    # Download file if not already present
    if not os.path.exists(archive_path):
        print(f"Downloading {filename}...")
        download_file(url, archive_path)
    
    # Extract the archive
    extract_to = os.path.join(download_root, 'extracted')
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    extract_archive(archive_path, extract_to)
    print(f"Extraction complete. Files are available in {extract_to}.")
