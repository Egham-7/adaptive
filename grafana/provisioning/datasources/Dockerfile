FROM grafana/grafana:latest

USER root
# Install gettext-base for envsubst

RUN apk add --no-cache gettext

COPY prometheus.prod.yml.tmpl /etc/grafana/provisioning/datasources/prometheus.yml.tmpl

# Create an entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER grafana
ENTRYPOINT ["entrypoint.sh"]
