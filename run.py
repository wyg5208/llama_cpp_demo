import uvicorn

from app.config import get_settings, setup_logging

if __name__ == "__main__":
    settings = get_settings()
    # Before uvicorn.run, and not only in app.main's lifespan: get_settings() above has
    # already logged a warning if it ignored a hand-edited settings_override.json, and
    # that is exactly the startup problem worth finding in the file later.
    #
    # uvicorn's configure_logging() leaves the root logger alone (disable_existing_loggers
    # is False and its LOGGING_CONFIG has no "root" key), so these handlers survive until
    # lifespan calls setup_logging again and replaces them. It does however re-install
    # uvicorn's own handler with propagate=False, undoing the reroute -- which is why the
    # two lines uvicorn logs before lifespan runs ("Started server process", "Waiting for
    # application startup.") reach the console but never the file. Measured, not assumed;
    # .env.example lists the six that do make it.
    setup_logging(settings)
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level="info")
