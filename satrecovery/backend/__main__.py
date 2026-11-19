"""Run the backend server::

    python -m satrecovery.backend                 # http://127.0.0.1:8000
    python -m satrecovery.backend --port 8100
    python -m satrecovery.backend --data-dir D:/satrec
    python -m satrecovery.backend --rml24 "C:/Users/me/Downloads/RML24_IQdata"

Then open http://127.0.0.1:8000/docs in a browser.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Satellite signal recovery backend")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="satrecovery_data",
                        help="where results and trained models are kept")
    parser.add_argument("--rml24", default=os.environ.get("SATREC_RML24"),
                        help="the RML24 dataset (RML24_IQdata.h5, its folder, or a sample .npz)")
    args = parser.parse_args()

    import uvicorn

    from .app import create_app
    from .settings import Settings

    settings = Settings(data_dir=args.data_dir, host=args.host, port=args.port, rml24_path=args.rml24)
    print(f"Data folder: {settings.data_dir}")
    if settings.rml24_path:
        print(f"Real recordings: {settings.rml24_path}")
    print(f"Open http://{settings.host}:{settings.port}/docs to try every endpoint.\n")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
