# IoT Data Platform POC

This project sets up a Proof of Concept (POC) platform for IoT data using Docker Compose. It incorporates Apache Iceberg, Trino, MinIO, and PostgreSQL to create a scalable and efficient data storage and query system.

## Components

- **MinIO**: Object storage with a web-based console for data inspection
- **PostgreSQL**: Database used as the Apache Iceberg metastore
- **Trino**: Distributed SQL query engine for reading and writing data
- **Jupyter Notebook**: For easy interaction with the data
- **Apache Iceberg**: Table format for huge analytic datasets

## Prerequisites

- Docker and Docker Compose installed on your system
- Basic understanding of Docker, SQL, and Python

## Setup Instructions

1. Clone this repository or create a new directory for the project.

2. Create the following directory structure:
   ```
   iot-data-platform/
   ├── docker-compose.yml
   ├── create_table.sql
   ├── trino-config/
   │   ├── config.properties
   │   └── catalog/
   │       ├── iceberg.properties
   │       └── minio.properties
   └── notebooks/
       └── iot_data_operations.ipynb
   ```

3. Copy the provided `docker-compose.yml`, `create_table.sql`, Trino configuration files, and Jupyter notebook into their respective directories.

4. Open a terminal, navigate to the `iot-data-platform` directory, and run:
   ```
   docker-compose up -d
   ```

5. Wait for all services to start. You can check the status with:
   ```
   docker-compose ps
   ```

6. Access the various services:
   - MinIO Console: http://localhost:9001 (login with minio/minio123)
   - Trino UI: http://localhost:8080
   - Jupyter Notebook: http://localhost:8888 (token will be in the container logs)

7. Create the Iceberg table by connecting to the Trino container and running the SQL script:
   ```
   docker exec -it iot-data-platform-trino-1 trino
   ```
   Then in the Trino CLI:
   ```sql
   USE iceberg.iot_data;
   SOURCE /path/to/create_table.sql;
   ```

8. Open the Jupyter notebook at http://localhost:8888 and navigate to the `iot_data_operations.ipynb` file to start interacting with your data.

## Using the Platform

1. **MinIO Console**: Use this to inspect raw data files and manage your object storage.

2. **Trino**: Use the Trino UI or CLI to run SQL queries against your Iceberg table.

3. **Jupyter Notebook**: Use the provided notebook to insert sample data and run queries using Python.

## Customization

- Modify the `create_table.sql` script to change the table schema as needed.
- Update the Trino configuration files in the `trino-config` directory to adjust settings.
- Extend the Jupyter notebook with additional data processing and analysis code.

## Troubleshooting

- If services fail to start, check the logs with `docker-compose logs [service_name]`.
- Ensure all ports specified in the `docker-compose.yml` file are available on your system.
- For Trino connection issues, verify the configuration files in the `trino-config` directory.

## Next Steps

- Implement data ingestion pipelines for your IoT devices.
- Develop data analysis and visualization tools using the Trino connection.
- Set up monitoring and alerting for your data platform.

## Support

For issues or questions, please open an issue in the project repository or contact the project maintainer.