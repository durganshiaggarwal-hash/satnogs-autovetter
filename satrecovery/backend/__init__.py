"""Phase 7 -- the backend: everything the project can do, over HTTP.

The GUI in Phase 8 talks to this, and so can anything else: a browser,
a script, a Jupyter notebook.  Start it with::

    python -m satrecovery.backend

and open http://127.0.0.1:8000/docs.

``settings``  where files live, and request size limits
``catalog``   the named receivers the API offers
``models``    training, saving and loading hybrid demappers
``jobs``      long work in the background, with progress
``schemas``   the request and response shapes
``services``  what each endpoint does, without any HTTP
``app``       the routes themselves
"""

from __future__ import annotations

from .settings import Settings


def create_app(settings: Settings | None = None):
    """Build the FastAPI application. Imported lazily so FastAPI is optional."""
    from .app import create_app as _create

    return _create(settings)


__all__ = ["Settings", "create_app"]
