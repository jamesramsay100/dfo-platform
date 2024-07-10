# Start up
```shell
docker-compose up -d
```

## [NOT USED] Spark setup
Create a table using demo nyc taxi data

Launch "Iceberg Getting Started notebook" via http://localhost:8888/
```shell
docker exec -it spark-iceberg notebook
```
Run notebook to create table `nyc.taxis` and insert data

Note that:
- the default catalogue is named `demo`
- database is named `nyc`
- table is named `taxis`

## Trino setup
Check `demo.properties` for Trino config. Note that the file name (i.e. demo) must match the catalogue name created by Spark

Open trino shell
```sh
docker-compose exec -it trino trino
```
Run some SQL commands:
```sql
SELECT 
    max(timestamp) 
FROM 
    demo.silver.dfo;
```
```sql
SELECT 
    distance as "Distance",
    avg(value) as "RMS0 average"
FROM 
    demo.silver.dfo 
WHERE
    distance < 10
AND
    timestamp >= now() - interval '5' minute
AND
    measurement = 'RMS0'
GROUP BY 
    distance
ORDER BY
    distance;
```