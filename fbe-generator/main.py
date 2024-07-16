import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from minio import Minio
from minio.error import S3Error
from io import BytesIO
from PIL import Image

# Minio client setup
minio_client = Minio(
    "minio:9000",
    access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    secure=False
)

# Environment variables
sensor_name = os.environ.get("SENSOR_NAME", "default_sensor")
asset_name = os.environ.get("ASSET_NAME", "default_asset")
well_name = os.environ.get("WELL_NAME", "default_well")
bucket_name = "dfo-bucket"
image_path = os.environ.get("IMAGE_PATH", "input_image.png")


# Load the image
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(img)


image_data = load_image(image_path)
image_height, image_width = image_data.shape


def generate_fbe(timestamp):
    df = pd.DataFrame({
        "Depth (m)": range(image_height),
    })

    time_index = timestamp % image_width

    for i in range(10):  # Generate 10 identical RMS columns
        df[f"RMS{i}"] = [image_data[d % image_height, time_index] for d in range(image_height)]

    print(f"Generated DataFrame with {len(df)} rows for timestamp {timestamp}")
    return df


def write_to_minio(df):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time}_{sensor_name}_{asset_name}_{well_name}.csv"

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    try:
        minio_client.put_object(
            bucket_name,
            filename,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type="application/csv"
        )
        print(f"Successfully wrote {filename} to MinIO")
    except S3Error as e:
        print(f"Error writing to MinIO: {e}")


def ensure_bucket_exists():
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully")
        else:
            print(f"Bucket '{bucket_name}' already exists")
    except S3Error as e:
        print(f"Error checking/creating bucket: {e}")
        raise


def main():
    start_time = int(time.time())
    next_time = start_time + 1  # Start of the next second

    while True:
        try:
            current_timestamp = int(time.time()) - start_time

            # Generate and write FBE
            df = generate_fbe(current_timestamp)
            write_to_minio(df)

            # Calculate sleep time
            current_time = time.time()
            sleep_time = next_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Processing took longer than 1 second (overrun: {-sleep_time:.3f}s)")

            # Set the next execution time
            next_time += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)  # Wait a bit before retrying


if __name__ == "__main__":
    ensure_bucket_exists()
    main()