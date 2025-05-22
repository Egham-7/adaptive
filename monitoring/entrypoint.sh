#!/bin/sh
envsubst </etc/prometheus/prometheus.yml.tmpl >/etc/prometheus/prometheus.yml
exec /bin/prometheus "$@"
