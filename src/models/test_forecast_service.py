from forecast_service import (
    ForecastService
)


service = ForecastService()


print("\nStores:")

print(
    service
    .get_available_stores()[:10]
)


print("\nFamilies:")

print(
    service
    .get_available_families()[:10]
)


print("\nForecast:")

result = (
    service.forecast_next_day(
        store_nbr=1,
        family="BEVERAGES",
        onpromotion=5,
    )
)

print(result)