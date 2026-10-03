from forecast_service import (
    ForecastService
)


service = ForecastService()


result = (
    service.forecast_multiple_days(
        store_nbr=1,
        family="BEVERAGES",
        forecast_days=7,
        promotions=[
            5,
            5,
            10,
            10,
            3,
            2,
            0,
        ],
    )
)


print()

print(
    "Store:",
    result["store_nbr"]
)

print(
    "Family:",
    result["family"]
)

print()

print(
    "FORECAST"
)

print(
    "-" * 50
)


for row in result["forecast"]:

    print(
        row["date"],
        " | ",
        row["predicted_sales"],
        "units",
        " | promo:",
        row["onpromotion"],
    )