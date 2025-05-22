FROM prom/prometheus:latest

USER root
COPY prometheus.prod.yml.tmpl /etc/prometheus/prometheus.yml.tmpl
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER nobody

EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]
