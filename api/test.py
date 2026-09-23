from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
record_dir = PROJECT_ROOT / "data" / "records500" / "00000"

with open(record_dir / "00001_hr.hea", "rb") as hea_file, open(
    record_dir / "00001_hr.dat", "rb"
) as dat_file:
    files = [
        ("files", ("00001_hr.hea", hea_file)),
        ("files", ("00001_hr.dat", dat_file)),
    ]
    response = requests.post("https://ecg-analysis-ws6t.onrender.com/predict", files=files, timeout=30)

response.raise_for_status()
print(response.json())
