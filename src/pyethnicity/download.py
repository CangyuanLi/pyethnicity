from pathlib import Path

import requests


def download():
    dat_path = Path(__file__).resolve().parents[0] / "data"
    r = requests.get(
        "https://raw.githubusercontent.com/CangyuanLi/pyethnicity/master/src/pyethnicity/data/models/first_sex.onnx"
    )
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(f"{r.status_code}: DOWNLOAD FAILED")

    with open(dat_path / "models/first_sex.onnx", "wb") as f:
        f.write(r.content)
