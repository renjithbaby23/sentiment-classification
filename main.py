import logging
import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from evaluate import loaded_model as model


class InputPayload(BaseModel):
    """
    Class defining the input payload format
    """
    content: str

MODEL_VERSION = "1.0.beta"
CLASS_NAMES = {0: "neutral", 1: "positive", 2: "negative"}

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "version": MODEL_VERSION,
            "message": str(exc),
            "data": None,
        },
    )


# predict endpoint
@app.post("/sentiment-detection/predict")
async def predict(payload: InputPayload):
    logging.debug("Received request")
    input_data = payload.content
    logging.debug("input data - %s", input_data)
    try:
        prediction = model(tf.constant([input_data]))
        result = {
            "raw text": input_data,
            "prediction confidence": str(np.max(prediction[0])),
            "decision": CLASS_NAMES[np.argmax(prediction[0])],
        }
        response = {
            "success": True,
            "version": MODEL_VERSION,
            "message": "sentiment prediction completed",
            "data": result,
        }
        return response
    except Exception as e:
        logging.error(e)
        error_response = {
            "success": False,
            "version": MODEL_VERSION,
            "message": str(e),
            "data": None,
        }
        return JSONResponse(status_code=500, content=error_response)


# health check endpoint
@app.get("/sentiment-detection/health")
async def health():
    sample_input = "How can I login to apple store online account?"
    try:
        prediction = model(tf.constant([sample_input]))
        result = {
            "raw text": sample_input,
            "prediction confidence": str(np.max(prediction[0])),
            "decision": CLASS_NAMES[np.argmax(prediction[0])],
        }
        response = {
            "success": True,
            "version": MODEL_VERSION,
            "message": "sentiment prediction health checked",
            "data": result,
        }
        return response
    except Exception as e:
        logging.error(e)
        error_response = {
            "success": False,
            "version": MODEL_VERSION,
            "message": str(e),
            "data": None,
        }
        return JSONResponse(status_code=500, content=error_response)
