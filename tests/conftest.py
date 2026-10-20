import pytest

from egcf_processing.config import CONFIG_ENV_VAR


@pytest.fixture(autouse=True)
def _no_repo_config(tmp_path_factory, monkeypatch):
    """Point every test at an empty egcf.toml so edits to the repo's copy can't change results."""
    path = tmp_path_factory.mktemp("config") / "egcf.toml"
    path.write_text("")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
