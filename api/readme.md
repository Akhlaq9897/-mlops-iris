This API uses FastAPI for serving predictions and Swagger UI for easy testing.

The trained model file (random_forest_model.pkl) must be stored in the models/ folder before running.

You can run the API locally via uvicorn or inside Jupyter Notebook using nest_asyncio.

Swagger UI is available at /docs and ReDoc at /redoc after the API starts.

Example features for the Iris dataset:

sepal_length: in cm (e.g., 5.1)

sepal_width: in cm (e.g., 3.5)

petal_length: in cm (e.g., 1.4)

petal_width: in cm (e.g., 0.2)

The prediction output is a class label (0, 1, or 2), corresponding to the Iris species.
