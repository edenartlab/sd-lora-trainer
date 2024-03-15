import os
import time
import subprocess

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest, '...')

    # Make sure the destination directory exists
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        subprocess.check_call(["wget", "-q", "-O", dest, url])
    except subprocess.CalledProcessError as e:
        print("Error occurred while downloading:")
        print("Exit status:", e.returncode)
        print("Output:", e.output)
    except Exception as e:
        print("An unexpected error occurred:", e)

    print(f"Downloading {url} took {time.time() - start} seconds")