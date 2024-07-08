FROM jupyter/pyspark-notebook:spark-3.3.2

USER root

# Install additional Python packages
RUN pip install trino pandas matplotlib

# Install Iceberg Spark Runtime
RUN wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.3_2.12/1.5.1/iceberg-spark-runtime-3.3_2.12-1.5.1.jar -O /usr/local/spark/jars/iceberg-spark-runtime-3.3_2.12-1.5.1.jar
RUN wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws/1.5.1/iceberg-aws-1.5.1.jar -O /usr/local/spark/jars/iceberg-aws-1.5.1.jar

# Install AWS Hadoop JARs for S3 access
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar -O /usr/local/spark/jars/hadoop-aws-3.3.2.jar && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.451/aws-java-sdk-bundle-1.12.451.jar -O /usr/local/spark/jars/aws-java-sdk-bundle-1.12.451.jar

RUN wget https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.17.247/url-connection-client-2.17.247.jar -O /usr/local/spark/jars/url-connection-client-2.17.247.jar

USER $NB_UID