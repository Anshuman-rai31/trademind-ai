"""TradeMind Application Launcher (app.py).

Forwards to run.py to support standard Flask entrypoint conventions.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = app.config.get("DEBUG", True)

    print("=" * 60)
    print(f" Starting TradeMind Platform ({app.config.get('PHASE', 'Phase 13')})")
    print(f" Environment : {os.environ.get('FLASK_ENV', 'development')}")
    print(f" Local URL   : http://{host}:{port}/")
    print(f" Health Check: http://{host}:{port}/health")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug, threaded=True)
