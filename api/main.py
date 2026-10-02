import sys
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
)

from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


sys.path.append(
    str(
        PROJECT_ROOT
        / "src"
        / "models"
    )
)


from forecast_service import (
    ForecastService
)

class MultiForecastRequest(
    BaseModel
):

    store_nbr: int = Field(
        ...,
        gt=0,
    )

    family: str = Field(
        ...,
        min_length=1,
    )

    forecast_days: int = Field(
        default=7,
        ge=1,
        le=30,
    )

    promotions: Optional[
        list[int]
    ] = None


app = FastAPI(

    title=(
        "FMCG Demand "
        "Forecasting API"
    ),

    description=(
        "Machine learning API "
        "for FMCG demand forecasting."
    ),

    version="1.0.0",
)


forecast_service = (
    ForecastService()
)


class ForecastRequest(
    BaseModel
):

    store_nbr: int = Field(
        ...,
        gt=0,
        description="Store number",
    )

    family: str = Field(
        ...,
        min_length=1,
        description=(
            "Product family"
        ),
    )

    onpromotion: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of products "
            "on promotion"
        ),
    )


class ForecastResponse(
    BaseModel
):

    store_nbr: int

    family: str

    forecast_date: str

    predicted_sales: float

    onpromotion: int


@app.get("/")
def root():

    return {
        "message":
            "FMCG Demand "
            "Forecasting API",

        "docs":
            "/docs",
    }


@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


@app.get("/stores")
def stores():

    return {
        "stores":
            forecast_service
            .get_available_stores()
    }


@app.get("/families")
def families():

    return {
        "families":
            forecast_service
            .get_available_families()
    }

@app.get("/model/info")
def model_info():

    return {

        "model":
            "LightGBM",

        "task":
            "FMCG demand forecasting",

        "forecast_type":
            "one-step-ahead",

        "target":
            "sales",

        "features":
            len(
                forecast_service
                .model
                .feature_name_
            ),
    }


@app.post(
    "/forecast",
    response_model=
        ForecastResponse,
)
def forecast(
    request: ForecastRequest
):

    try:

        result = (
            forecast_service
            .forecast_next_day(

                store_nbr=
                    request.store_nbr,

                family=
                    request.family,

                onpromotion=
                    request.onpromotion,
            )
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecasting failed: "
                f"{str(exc)}"
            ),
        )

@app.post(
    "/forecast/multi"
)
def multi_forecast(
    request: MultiForecastRequest
):

    try:

        result = (
            forecast_service
            .forecast_multiple_days(

                store_nbr=
                    request.store_nbr,

                family=
                    request.family,

                forecast_days=
                    request.forecast_days,

                promotions=
                    request.promotions,
            )
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecasting failed: "
                f"{str(exc)}"
            ),
        )