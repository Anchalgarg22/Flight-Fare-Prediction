FROM tiangolo/uwsgi-nginx-flask:python3.8
WORKDIR /app/
COPY flightprediction.pkl .
COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY app.py  /app/
CMD ["python", "app.py"]