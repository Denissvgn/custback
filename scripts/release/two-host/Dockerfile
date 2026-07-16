FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        iptables \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG CUSTBACK_WHEEL
COPY ${CUSTBACK_WHEEL} /tmp/${CUSTBACK_WHEEL}
RUN python -m pip install /tmp/${CUSTBACK_WHEEL} \
    && rm -f /tmp/${CUSTBACK_WHEEL}

COPY probe.py /opt/custback-two-host-probe.py

CMD ["python", "-c", "import time; time.sleep(86400)"]
