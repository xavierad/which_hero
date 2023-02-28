FROM  python:3.10-slim-bullseye

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y gcc \
        libpq-dev\
        python-dev

WORKDIR /web_app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

COPY . .