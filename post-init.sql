CREATE SCHEMA example.silver WITH (location = 's3://warehouse');

CREATE TABLE example.silver.F_SENSOR_DATA_DISTRIBUTED
(
  timestamp TIMESTAMP,
  sensor_key VARCHAR,
  measurement_key VARCHAR,
  distance DOUBLE,
  value DOUBLE
)
WITH (
  format = 'PARQUET'
);

INSERT INTO example.silver.F_SENSOR_DATA_DISTRIBUTED
(timestamp, sensor_key, measurement_key, distance, value)
VALUES
(TIMESTAMP '2023-07-04 10:30:00', 'SENSOR001', 'TEMPERATURE', 5.2, 23.5),
(TIMESTAMP '2023-07-04 10:31:00', 'SENSOR002', 'HUMIDITY', 3.8, 65.2),
(TIMESTAMP '2023-07-04 10:32:00', 'SENSOR003', 'PRESSURE', 2.1, 1013.25),
(TIMESTAMP '2023-07-04 10:33:00', 'SENSOR001', 'TEMPERATURE', 5.2, 23.7),
(TIMESTAMP '2023-07-04 10:34:00', 'SENSOR002', 'HUMIDITY', 3.8, 65.5);