from datetime import datetime
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from fastapi import APIRouter, Form, HTTPException, UploadFile
from schemas.model import ModelBaseSchema, ModelReadSchema
from services.model_service import (
    CustomModelService,
    HuggingFaceModelService,
    ModelService,
)
from sqlalchemy.orm import Session

model_router = APIRouter(prefix="/models", tags=["Models"])


# TODO: 책임 분리 필요.
@model_router.post("", response_model=ModelReadSchema)
def create_model(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    model_provider_id: Annotated[int, Form()],
    model_type_id: Annotated[int, Form()],
    model_format_id: Annotated[int, Form()],
    file: UploadFile | None = None,
):
    """
    Model Registry에 Model을 등록하는 API

    * params
        - model_provider
            1. huggingface
            2. ollama
            3. custom
        - model_type
            1. llm
            2. embedding model
            3. re rank
            4. fine tuned
        - model_format
            1. transformers
            2. sentence-transformers
            3. gguf
            4. bge-m3
    """

    model = ModelBaseSchema(
        name=name,
        description=description,
        model_provider_id=model_provider_id,
        model_type_id=model_type_id,
        model_format_id=model_format_id,
    )

    try:
        if model_provider_id == 1:  # HuggingFace
            result = HuggingFaceModelService().create(model, db)
        elif model_provider_id == 2:  # Ollama
            ...
        elif model_provider_id == 3:  # Custom
            result = CustomModelService().create(model, file, db)
        else:
            print("Error has been occured!")
        return result
    except Exception as e:
        db.rollback()
        raise e


@model_router.get("/{model_id}", response_model=ModelReadSchema)
def read_model(model_id: int, db: Session = SessionDepends):
    db_model = ModelService().get(db, model_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@model_router.get("", response_model=list[ModelReadSchema])
def read_models(skip: int = 0, limit: int = 10, db: Session = SessionDepends):
    models = ModelService().get_multi(db)
    return models


@model_router.get("/{model_id}/test")
def test_models(model_id: int, model_format_id: int, model_uri: str):
    result = ModelService().validate(model_format_id, model_uri)
    return model_uri
