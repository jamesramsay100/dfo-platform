#!/bin/bash

#nohup /usr/lib/trino/bin/run-trino &
#
#sleep 10

echo "Beginning initialisation..."

trino < /tmp/post-init.sql

echo "Tables and schemas initialised..."