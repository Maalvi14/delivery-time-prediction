-- Top 5 customer areas with highest average delivery time in the last 30 days
SELECT
  d.customer_area,
  AVG(d.delivery_time_min)        AS avg_delivery_time_min,
  COUNT(*)                        AS deliveries
FROM deliveries AS d
WHERE d.order_placed_at >= CURRENT_DATE - INTERVAL '30 days'
  AND d.customer_area IS NOT NULL
GROUP BY d.customer_area
ORDER BY avg_delivery_time_min DESC
LIMIT 5;

-- Average delivery time per traffic condition, by restaurant area and cuisine type
SELECT
  r.area                         AS restaurant_area,
  r.cuisine_type,
  d.traffic_condition,
  AVG(d.delivery_time_min)       AS avg_delivery_time_min,
  COUNT(*)                       AS deliveries
FROM deliveries AS d
JOIN orders      AS o ON o.delivery_id     = d.delivery_id
JOIN restaurants AS r ON r.restaurant_id   = o.restaurant_id
GROUP BY r.area, r.cuisine_type, d.traffic_condition
ORDER BY r.area, r.cuisine_type, d.traffic_condition;

-- Top 10 delivery people with fastest average delivery time, only active and with ≥ 50 deliveries
WITH per_person AS (
  SELECT
    CAST(d.delivery_person_id AS INT) AS delivery_person_id,
    AVG(d.delivery_time_min)          AS avg_delivery_time_min,
    COUNT(*)                          AS deliveries_count
  FROM deliveries AS d
  GROUP BY CAST(d.delivery_person_id AS INT)
)
SELECT
  dp.delivery_person_id,
  dp.name,
  dp.region,
  p.deliveries_count,
  p.avg_delivery_time_min
FROM per_person       AS p
JOIN delivery_persons AS dp
  ON dp.delivery_person_id = p.delivery_person_id
WHERE dp.is_active = TRUE
  AND p.deliveries_count >= 50
ORDER BY p.avg_delivery_time_min ASC
LIMIT 10;

-- Most profitable restaurant area in the last 3 months
SELECT
  r.area                          AS restaurant_area,
  SUM(o.order_value)              AS total_order_value,
  COUNT(*)                        AS orders_count
FROM orders      AS o
JOIN deliveries  AS d ON d.delivery_id     = o.delivery_id
JOIN restaurants AS r ON r.restaurant_id   = o.restaurant_id
WHERE d.order_placed_at >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY r.area
ORDER BY total_order_value DESC
LIMIT 1;

-- Increasing trend in average delivery time (last 12 weeks, weekly buckets)
WITH daily AS (
  SELECT
    CAST(d.delivery_person_id AS INT)        AS delivery_person_id,
    DATE_TRUNC('day', d.order_placed_at)     AS day,
    AVG(d.delivery_time_min)                 AS avg_delivery_time_min
  FROM deliveries AS d
  GROUP BY CAST(d.delivery_person_id AS INT),
           DATE_TRUNC('day', d.order_placed_at)
),
slopes AS (
  SELECT
    delivery_person_id,
    /* slope in minutes per day */
    REGR_SLOPE(
      avg_delivery_time_min,
      EXTRACT(EPOCH FROM day)::BIGINT / 86400.0
    )                                        AS slope_min_per_day,
    REGR_R2(
      avg_delivery_time_min,
      EXTRACT(EPOCH FROM day)::BIGINT / 86400.0
    )                                        AS r2,
    COUNT(*)                                  AS n_days
  FROM daily
  GROUP BY delivery_person_id
)
SELECT
  s.delivery_person_id,
  dp.name,
  dp.region,
  s.n_days,
  s.slope_min_per_day,
  s.r2
FROM slopes AS s
JOIN delivery_persons AS dp
  ON dp.delivery_person_id = s.delivery_person_id
WHERE s.n_days >= 14          -- ensure enough history
  AND s.slope_min_per_day > 0 -- increasing trend
ORDER BY s.slope_min_per_day DESC;