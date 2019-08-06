FROM python:3.6

LABEL Name=citi-bikecaster-train

WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app/requirements.txt

RUN python3 -m pip install -r requirements.txt

ADD . /usr/src/app

