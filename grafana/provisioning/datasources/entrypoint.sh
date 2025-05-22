#!/bin/bash

# Replace environment variables in the datasource configuration
envsubst < /etc/grafana/provisioning/datasources/prometheus.yml.tmpl > /etc/grafana/provisioning/datasources/prometheus.yml

# Start Grafana
exec /run.sh
