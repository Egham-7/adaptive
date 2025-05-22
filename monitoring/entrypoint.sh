#!/bin/bash

# Replace environment variables in the configuration
envsubst < /etc/prometheus/prometheus.yml.tmpl > /etc/prometheus/prometheus.yml

# Start Prometheus with the generated config
exec /bin/prometheus --config.file=/etc/prometheus/prometheus.yml
