# Trino Playground with Iceberg

Init command:
```sh
docker-compose up -d
```
```sh
docker-compose exec trino sh -c "chmod +x /tmp/post-init.sh && /tmp/post-init.sh"
```

Enter Trino shell:
```sh
docker-compose exec -it trino trino
```

Then, you can go.

For example:
```
SELECT * FROM example.silver.F_SENSOR_DATA_DISTRIBUTED;
```

## Existing Settings

- Catalog: example
  - Schema: example_s3_schema
    - Table: employees_test

Catalog is created in [example.properties](./example.properties).  
Schema and table are created in [post-init.sql](./post-init.sql).