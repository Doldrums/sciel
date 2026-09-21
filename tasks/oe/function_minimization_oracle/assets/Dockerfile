ARG PYTHON_IMAGE=python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
FROM ${PYTHON_IMAGE}
WORKDIR /opt/oracle
COPY container_supervisor.py oracle_client.py candidate_runner.py ./
RUN chmod 0555 /opt/oracle && chmod 0444 /opt/oracle/*.py && mkdir /input
ENTRYPOINT ["python", "-I", "-B", "/opt/oracle/container_supervisor.py"]
