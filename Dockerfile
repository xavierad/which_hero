FROM  python:3.6.13

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y gcc \
        libpq-dev\
        python-dev

WORKDIR /which_hero

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

COPY . .