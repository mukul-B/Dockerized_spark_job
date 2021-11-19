FROM python:3.6.7

ADD main.py .
RUN #export PIP_DEFAULT_TIMEOUT=100
RUN pip install pyspark

CMD ["python", "./m2.py"]