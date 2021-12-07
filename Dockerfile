#FROM openjdk:11
FROM python:3.8.7

#ENV JAVA_HOME /usr/java/openjdk-11
#ENV PATH $JAVA_HOME/bin:$PATH
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    rm -rf /var/cache/oracle-jdk8-installer;

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

ADD main.py .
RUN #export PIP_DEFAULT_TIMEOUT=100
RUN pip install pyspark
RUN pip install numpy
ADD winequality-white.csv .

#CMD ["python", "./main.py"]
COPY ./entry_point.sh /
RUN chmod 775 /entry_point.sh
ENTRYPOINT ["/entry_point.sh"]