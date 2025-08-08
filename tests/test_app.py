from fastapi.testclient import TestClient
import importlib
import sys
import types


class _DummyModel:
    def predict(self, features):
        return [1]


def test_predict_endpoint_works():
    # Stub joblib before importing the app module to avoid loading a real file
    sys.modules["joblib"] = types.SimpleNamespace(load=lambda _path: _DummyModel())

    app_module = importlib.import_module("api.app")
    client = TestClient(app_module.app)

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)

