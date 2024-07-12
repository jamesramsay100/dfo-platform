import os

# Use a directory that Superset can write to
SQLALCHEMY_DATABASE_URI = 'sqlite:////app/superset_home/superset.db'

# Add Trino database connection
DATABASES = {
    'trino': {
        'allow_csv_upload': False,
        'allow_ctas': False,
        'allow_cvas': False,
        'database_name': 'Trino',
        'extra': {
            'engine_params': {
                'connect_args': {
                    'protocol': 'https',
                    'verify': False,
                }
            }
        },
        'sqlalchemy_uri': 'trino://trino:8080/demo'
    }
}

# Set the default database
DEFAULT_DB_ID = 'trino'

# Set a secret key
SECRET_KEY = 'insecure_key_for_dev'