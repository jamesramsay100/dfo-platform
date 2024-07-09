```shell
docker-compose up -d
```


## Spark setup
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
    count(*) 
FROM 
    demo.nyc.taxis;
```
```sql
SELECT 
    payment_type as "Payment Type",
    avg(passenger_count) as "Average passenger count"
FROM 
    demo.nyc.taxis 
GROUP BY 
    payment_type;
```