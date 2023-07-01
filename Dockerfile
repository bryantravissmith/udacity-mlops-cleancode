FROM python:3.8

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888  --allow-root
