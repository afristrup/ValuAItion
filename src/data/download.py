import os
import re
import wget
import logging
import requests
from pathlib import Path
from .constants import DATA_DIR, BASE_FILENAME, BASE_DATASET_URL


def download_data():
    for split in ["TRAIN", "TEST"]:
        filename = f"{BASE_FILENAME}_{split}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(filepath):
            logging.info(
                "Downloading dataset file: %s from %s", filename, BASE_DATASET_URL
            )
            filename = wget.download(f"{BASE_DATASET_URL}/{filename}", out=filepath)
        else:
            logging.debug(
                "Skipping download of already existing dataset file: %s", filepath
            )


def download_instreg():
    """Download InstReg CSV file from statistik.uni-c.dk"""
    url = "https://statistik.uni-c.dk/instregudtraek/"
    filename = "filename=InstReg-udtraek17-01-2026.csv"
    filepath = os.path.join(DATA_DIR, filename)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    if os.path.exists(filepath):
        logging.info(f"InstReg file already exists: {filepath}")
        return

    logging.info(f"Downloading InstReg file from {url}")
    try:
        # Create a session to maintain cookies
        session = requests.Session()

        # First, GET the page to extract ViewState and EventValidation
        logging.info("Fetching form page to extract ViewState and EventValidation...")
        get_response = session.get(url, timeout=30)
        get_response.raise_for_status()

        # Parse HTML to extract ViewState and EventValidation using regex
        html_content = get_response.text

        # Extract __VIEWSTATE (handle multiline with re.DOTALL)
        viewstate_match = re.search(
            r'<input[^>]*name=["\']__VIEWSTATE["\'][^>]*value=["\']([^"\']*)["\']',
            html_content,
            re.DOTALL,
        )
        viewstate = viewstate_match.group(1) if viewstate_match else ""

        # Extract __VIEWSTATEGENERATOR
        viewstate_gen_match = re.search(
            r'<input[^>]*name=["\']__VIEWSTATEGENERATOR["\'][^>]*value=["\']([^"\']*)["\']',
            html_content,
            re.DOTALL,
        )
        viewstate_generator = (
            viewstate_gen_match.group(1) if viewstate_gen_match else ""
        )

        # Extract __EVENTVALIDATION
        eventval_match = re.search(
            r'<input[^>]*name=["\']__EVENTVALIDATION["\'][^>]*value=["\']([^"\']*)["\']',
            html_content,
            re.DOTALL,
        )
        event_validation = eventval_match.group(1) if eventval_match else ""

        if not viewstate or not event_validation:
            raise ValueError(
                "Could not find required form fields (ViewState/EventValidation)"
            )

        # Prepare form data to submit the CSV download request
        # Select "all" for both institution type and municipality (value="-1")
        form_data = {
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": viewstate_generator,
            "__EVENTVALIDATION": event_validation,
            "ctl00$MainContent$InstType3ListBox": "-1",  # Select all institution types
            "ctl00$MainContent$KommuneListBox": "-1",  # Select all municipalities
            "ctl00$MainContent$UdtrækButton": "Dan udtræk til CSV-format",  # Click the CSV download button
        }

        # POST request to submit the form and download the CSV
        logging.info("Submitting form to download CSV...")
        response = session.post(url, data=form_data, timeout=60)
        response.raise_for_status()

        # Check if we got HTML instead of CSV (error case)
        if response.text.strip().startswith(
            "<!DOCTYPE"
        ) or response.text.strip().startswith("<html"):
            raise ValueError(
                "Received HTML instead of CSV. The form submission may have failed."
            )

        # Save the file (response should be CSV, possibly UTF-16 encoded)
        with open(filepath, "wb") as f:
            f.write(response.content)

        logging.info(f"Successfully downloaded InstReg file to {filepath}")
    except Exception as e:
        logging.error(f"Failed to download InstReg file: {e}")
        raise


if __name__ == "__main__":
    import sys

    logging.getLogger().setLevel(logging.DEBUG)

    if len(sys.argv) > 1 and sys.argv[1] == "--instreg":
        download_instreg()
    else:
        download_data()
