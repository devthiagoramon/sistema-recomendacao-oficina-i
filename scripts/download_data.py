"""Baixa e extrai o MovieLens (ml-latest-small) em data/."""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST = Path(__file__).resolve().parents[1] / "data"


def main():
    target = DEST / "ml-latest-small"
    if (target / "ratings.csv").exists():
        print(f"Dataset já existe em {target}")
        return
    DEST.mkdir(exist_ok=True)
    print(f"Baixando {URL} ...")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(URL, timeout=60, context=ctx) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DEST)
    print(f"Pronto: {target}")


if __name__ == "__main__":
    main()
