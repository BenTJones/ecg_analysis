import os
import tempfile
from typing import List

import wfdb
from fastapi import FastAPI, File, HTTPException, UploadFile

from api.predict import predict

app = FastAPI(title="ECG Risk Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_ecg(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one WFDB file (.hea and .dat).")

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            file_path = os.path.join(tmpdir, upload.filename)
            with open(file_path, "wb") as handle:
                handle.write(await upload.read())

        hea_files = [name for name in os.listdir(tmpdir) if name.endswith(".hea")]
        if not hea_files:
            raise HTTPException(status_code=400, detail="Upload must include a .hea header file.")

        record_path = os.path.join(tmpdir, hea_files[0].replace(".hea", ""))
        try:
            signal, metadata = wfdb.rdsamp(record_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read WFDB record. Upload both .hea and .dat files. Error: {exc}",
            ) from exc

    return predict(signal, metadata["fs"])
