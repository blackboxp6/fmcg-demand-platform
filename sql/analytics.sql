SELECT
    family,
    SUM(sales) AS total_sales,
    AVG(sales) AS avg_daily_sales
FROM
    'data/gold/forecast_features.parquet'
GROUP BY
    family
ORDER BY
    total_sales DESC
LIMIT 20;