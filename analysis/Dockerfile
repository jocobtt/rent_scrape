FROM python:3.10-slim as builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /work
COPY requirements.txt .
COPY model_api.py .
COPY passed-model.joblib .
COPY challenger-model.joblib .

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# final stage
FROM python:3.10-slim
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /work /work

WORKDIR /work
ENV PATH="/opt/venv/bin:$PATH"
CMD ["python3", "model_api.py"]
