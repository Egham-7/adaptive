FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir 'litgpt[extra]' \
  transformers \
  sentencepiece

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV LITGPT_SERVE_PORT="8000"
ENV MODEL_ID="microsoft/phi-2" 

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
