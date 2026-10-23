
"""
Download NUCLE (NUS Corpus of Learner English)

Source (verified via web search, 2026-09-24):
    https://huggingface.co/datasets/nusnlp/NUCLE
    -> data/nucle3.2.sgml (source filename; release must be verified separately)

Licensing:
    Distributed under the "standard NUS licensing agreement"
    (see https://huggingface.co/datasets/nusnlp/NUCLE/blob/main/nucle_license.pdf).
    By downloading this file you agree to that license. Read it and cite:
    Dahlmeier, Ng & Wu (2013), "Building a Large Annotated Corpus of Learner
    English: The NUS Corpus of Learner English", BEA 2013.
    https://aclanthology.org/W13-1703/

Format (per the official dataset card):
    <DOC nid="840">
      <TEXT><P>...</P>...</TEXT>
      <ANNOTATION teacher_id="173">
        <MISTAKE start_par="0" start_off="0" end_par="0" end_off="26">
          <TYPE>ArtOrDet</TYPE>
          <CORRECTION>The engineering design process</CORRECTION>
        </MISTAKE>
        ...
      </ANNOTATION>
    </DOC>
    ...

This module downloads the SGML only. Parsing is handled by nucle_sgml.py.
"""

import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 60  # seconds -- file is ~12.6 MB
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3

URL = "https://huggingface.co/datasets/nusnlp/NUCLE/resolve/main/data/nucle3.2.sgml?download=true"

if __package__:
    from .constants import SGML_FILE
else:
    from constants import SGML_FILE

RAW_DIR = SGML_FILE.parent


class DownloadData:
    @staticmethod
    def fetch_data(url: str, dest: Path) -> None:
        """Stream-download `url` to `dest`, creating parent dirs as needed."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            print(f"Already downloaded, skipping: {dest}")
            return

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with requests.get(
                    url,
                    stream=True,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                print(f"Saved: {dest}")
                return

            except requests.exceptions.RequestException as exc:
                last_error = exc
                print(f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if dest.exists():
                    dest.unlink()  # remove partial file before retrying
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"Could not download {url} after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}. "
            "If the file is gated behind a Hugging Face login, download it "
            "manually from https://huggingface.co/datasets/nusnlp/NUCLE "
            "(accept the license there) and place it at "
            f"{dest}, then re-run this script -- it will skip the download "
            "step since the file already exists."
        ) from last_error

if __name__ == "__main__":
    DownloadData.fetch_data(URL, SGML_FILE)
