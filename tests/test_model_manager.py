from churn import model_manager


def test_ensure_model_does_nothing_when_model_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "churn_pipeline.joblib"
    model_path.write_text("fake model")

    monkeypatch.setattr(model_manager, "MODEL_PATH", model_path)

    def fail_download():
        raise AssertionError("download_model should not be called")

    monkeypatch.setattr(model_manager, "download_model", fail_download)

    model_manager.ensure_model()


def test_ensure_model_downloads_when_model_missing(tmp_path, monkeypatch):
    model_path = tmp_path / "churn_pipeline.joblib"

    monkeypatch.setattr(model_manager, "MODEL_PATH", model_path)

    called = False

    def fake_download():
        nonlocal called
        called = True
        model_path.write_text("fake model")

    monkeypatch.setattr(model_manager, "download_model", fake_download)

    model_manager.ensure_model()

    assert called
    assert model_path.exists()
