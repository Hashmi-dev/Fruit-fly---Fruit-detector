# get_data_light.py
# One-off helper: downloads ONE copy of the fruit images straight out of the Kaggle zip.
# Why it exists: the zip is 3.85 GB because it holds the dataset twice, and disk space is tight.
# This script reads single files out of the remote zip, so the zip itself is never saved.
# Run it once from the project folder:   python get_data_light.py
# (Notebook 00 can also unpack a normal zip if you would rather upload one.)

import io  # gives us the base class for a file-like object
import os  # for building folder paths
import zipfile  # reads the zip format
import urllib.request  # downloads bytes over HTTP
from concurrent.futures import ThreadPoolExecutor  # runs several downloads at once

# public download address of the dataset (no login needed)
URL = "https://www.kaggle.com/api/v1/datasets/download/sriramr/fruits-fresh-and-rotten-for-classification"
# where the images go, relative to the project folder
OUT_DIR = os.path.join("data", "raw")


class RemoteFile(io.RawIOBase):
    """A 'file' that lives on a web server and is read piece by piece."""

    def __init__(self, url):
        # ask for 1 byte so the server tells us the final address and the total size
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request) as response:
            self.url = response.geturl()  # address after Kaggle redirects us to storage
            self.size = int(response.headers["Content-Range"].split("/")[1])  # total bytes
        self.pos = 0  # our current reading position in the remote file

    def seekable(self):
        return True  # tells zipfile we can jump around

    def readable(self):
        return True  # tells zipfile we can read

    def tell(self):
        return self.pos  # report the current position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset  # jump to an absolute position
        elif whence == 1:
            self.pos = self.pos + offset  # jump relative to where we are
        else:
            self.pos = self.size + offset  # jump relative to the end
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0  # nothing left to read
        last = min(self.pos + len(buffer), self.size) - 1  # last byte we want
        data = fetch_range(self.url, self.pos, last)  # download just those bytes
        buffer[: len(data)] = data  # copy them into the caller's buffer
        self.pos = self.pos + len(data)  # move our position forward
        return len(data)


def fetch_range(url, first, last):
    """Download bytes first..last (inclusive) from url, retrying a few times."""
    for attempt in range(5):  # network hiccups happen, so try up to 5 times
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as error:
            last_error = error  # remember the error in case every attempt fails
    raise last_error


def main():
    remote = RemoteFile(URL)  # open the remote zip
    print("remote zip size (GB):", round(remote.size / 1e9, 2))
    # the buffer makes zipfile read the index in a few big requests instead of many tiny ones
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=4 * 1024 * 1024))

    wanted = []  # list of (zip entry, path on disk) for the copy we keep
    for info in archive.infolist():
        parts = info.filename.split("/")  # e.g. ['dataset', 'train', 'freshapples', 'x.png']
        # keep only dataset/<split>/<class>/<file>; this skips the duplicate dataset/dataset/...
        if len(parts) != 4 or parts[0] != "dataset" or parts[1] not in ("train", "test"):
            continue
        if info.is_dir() or parts[3] == "":
            continue  # skip folder entries
        # build the path from known-safe pieces only (basename stops odd paths in the zip)
        target = os.path.join(OUT_DIR, parts[1], os.path.basename(parts[2]), os.path.basename(parts[3]))
        wanted.append((info, target))

    total_mb = sum(info.compress_size for info, target in wanted) / 1e6
    print("files to download:", len(wanted), " size (MB):", round(total_mb))

    def download_one(item):
        info, target = item
        if os.path.exists(target) and os.path.getsize(target) == info.file_size:
            return 0  # already downloaded, so the script can be re-run safely
        os.makedirs(os.path.dirname(target), exist_ok=True)  # make the class folder
        # each zip entry = a small header, then the (compressed) file bytes
        header_and_data = fetch_range(
            remote.url, info.header_offset, info.header_offset + 30 + 600 + info.compress_size
        )  # 30 = fixed header size, 600 = generous room for the file name stored in the header
        name_len = int.from_bytes(header_and_data[26:28], "little")  # length of the stored name
        extra_len = int.from_bytes(header_and_data[28:30], "little")  # length of extra info
        start = 30 + name_len + extra_len  # where the real file bytes begin
        raw = header_and_data[start : start + info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            data = raw  # stored without compression
        else:
            import zlib  # only needed for compressed entries

            data = zlib.decompress(raw, -15)  # -15 = raw deflate stream, as used inside zips
        if zipfile.crc32(data) != info.CRC:
            raise ValueError("checksum mismatch for " + info.filename)  # never keep a corrupt file
        with open(target, "wb") as f:
            f.write(data)  # save the image
        return 1

    done = 0  # how many files we have processed
    with ThreadPoolExecutor(max_workers=16) as pool:  # 16 downloads in parallel
        for result in pool.map(download_one, wanted):
            done = done + 1
            if done % 1000 == 0:
                print("processed", done, "of", len(wanted), flush=True)
    print("finished. images are in", OUT_DIR)


if __name__ == "__main__":
    main()
