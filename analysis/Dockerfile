FROM python:3.10-slim as builder 

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN mkdir /work
WORKDIR /work
COPY requirements.txt requirements.txt
COPY model_api.py work/model_api.py
COPY passed-model work/passed-model 
COPY passed-model.joblib work/passed-model.joblib
COPY challenger_model.joblib work/challenger_model.joblib

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# final stage 
FROM python:3.10-slim
RUN mkdir /work
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /work .
WORKDIR /work

ENV PATH="/opt/venv/bin:$PATH"
CMD ["python3", "model_api.py"]
