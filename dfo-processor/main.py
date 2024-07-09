import os
import time
from pyiceberg.catalog import load_catalog
from pyiceberg.types import TimestampType, StringType, LongType, DoubleType
from pyiceberg.schema import Schema
from pyiceberg.types import (
    TimestampType,
    StringType,
    LongType,
    DoubleType,
    NestedField
)
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform, IdentityTransform
from pyiceberg.table.sorting import SortOrder, SortField
import pandas as pd
import pyarrow as pa
from datetime import datetime
from minio import Minio
from minio.commonconfig import REPLACE, CopySource
from minio.error import S3Error

print("Script started")

# MinIO client setup
minio_client = Minio(
    "minio:9000",
    access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    secure=False
)
print("Connected to minio")

# Iceberg catalog setup
catalog = load_catalog(
    "demo",
    **{
        "type": "rest",
        "uri": "http://rest:8181",
        "warehouse": "s3://warehouse/wh/",
        "s3.endpoint": "http://minio:9000",
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "s3.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "s3.path-style-access": "true",
    }
)
print("Loaded catalogue")

schema = Schema(
    NestedField(field_id=1, name="timestamp", field_type=TimestampType(), required=True),
    NestedField(field_id=2, name="sensor_name", field_type=StringType(), required=True),
    NestedField(field_id=3, name="distance", field_type=LongType(), required=True),
    NestedField(field_id=4, name="measurement", field_type=StringType(), required=True),
    NestedField(field_id=5, name="value", field_type=DoubleType(), required=True)
)
pa_schema = pa.schema([
    pa.field("timestamp", pa.timestamp('us'), nullable=False),
    pa.field("sensor_name", pa.string(), nullable=False),
    pa.field("distance", pa.int64(), nullable=False),
    pa.field("measurement", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=False)
])

try:
    catalog.create_namespace("silver")
    print("Namespace 'silver' created successfully.")
except Exception as e:
    print(f"Error creating namespace: {e}")

try:
    catalog.create_table("silver.dfo", schema=schema)
except Exception as e:
    print(f"Error creating table: {e}")

table = catalog.load_table("silver.dfo")


def ensure_bucket_exists(bucket_name):
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully")
        else:
            print(f"Bucket '{bucket_name}' already exists")
    except S3Error as e:
        print(f"Error checking/creating bucket: {e}")
        raise


def process_csv(object_name):
    # Download CSV from MinIO
    response = minio_client.get_object("dfo-bucket", object_name)
    print(f"loaded minio file {response}")
    df = pd.read_csv(response)

    # Extract timestamp and sensor_name from filename
    filename = object_name.split("/")[-1]
    filename_parts = filename.split("_")

    if len(filename_parts) >= 2:
        timestamp_str = "_".join(filename_parts[:2])  # Join first two parts for timestamp
        sensor_name = filename_parts[2] if len(filename_parts) > 2 else "unknown"
    else:
        print(f"Warning: Unexpected filename format: {filename}")
        timestamp_str = "19700101_000000"  # Use a default timestamp
        sensor_name = "unknown"

    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        print(f"Warning: Unable to parse timestamp from filename: {filename}")
        timestamp = datetime.utcfromtimestamp(0)  # Use Unix epoch as default

    # Rename "Depth (m)" to "distance"
    df = df.rename(columns={"Depth (m)": "distance"})

    # Melt RMS columns
    id_vars = ["distance"]
    value_vars = [col for col in df.columns if col.startswith("RMS")]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="measurement", value_name="value")

    # Add timestamp and sensor_name columns
    df_melted["timestamp"] = timestamp
    df_melted["sensor_name"] = sensor_name

    # Reorder columns to match schema
    df_final = df_melted[["timestamp", "sensor_name", "distance", "measurement", "value"]]

    return df_final


def move_to_archive(object_name):
    try:
        # Copy object to archive bucket
        result = minio_client.copy_object(
            "dfo-archive",
            object_name,
            CopySource("dfo-bucket", object_name)
        )

        # Remove object from original bucket
        minio_client.remove_object("dfo-bucket", object_name)

        print(f"Moved {object_name} to archive bucket")
    except S3Error as e:
        print(f"Error moving file to archive: {e}")
        raise


def main():
    ensure_bucket_exists("dfo-archive")

    while True:
        try:
            new_files = False
            for obj in minio_client.list_objects("dfo-bucket", recursive=True):
                if obj.object_name.endswith(".csv"):
                    print(f"Processing {obj.object_name}")
                    df = process_csv(obj.object_name)

                    # Ensure correct data types
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['distance'] = df['distance'].astype('int64')
                    df['value'] = df['value'].astype('float64')

                    print(df.head())
                    print(df.dtypes)

                    # Convert to PyArrow table
                    tb = pa.Table.from_pandas(df, schema=pa_schema)
                    print("converted to pa")

                    # Append to Iceberg table
                    table.append(tb)

                    # Move processed file to archive
                    move_to_archive(obj.object_name)

                    new_files = True

            if new_files:
                print("Batch processing complete. Sleeping for 5 seconds.")
            else:
                print("No new files found. Sleeping for 5 seconds.")

            time.sleep(5)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)  # Sleep for 5 seconds before retrying in case of an error


if __name__ == "__main__":
    main()