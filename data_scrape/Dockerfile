FROM python:3.10-slim as builder 

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN mkdir /work
WORKDIR /work
COPY requirements.txt /work
COPY bs4-rent.py bs4-rent.py

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# final stage 
FROM python:3.10-slim
COPY --from=builder /opt/venv /opt/venv
WORKDIR /work

ENV PATH="/opt/venv/bin:$PATH"
CMD ["python3", "bs4-rent.py"]
