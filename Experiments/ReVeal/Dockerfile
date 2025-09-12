FROM ubuntu:20.04

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get install -y unzip

RUN apt-get install -y python3-venv && apt-get install -y python3-dev && apt-get install -y git && apt-get install -y build-essential && apt-get install -y python3-distutils  && apt-get install -y python3-setuptools
RUN apt-get install -y python3-pip && pip install --upgrade pip && apt-get install -y libgraphviz-dev &&  pip install pygraphviz
RUN wget https://services.gradle.org/distributions/gradle-2.0-all.zip && unzip gradle-2.0-all.zip
RUN apt-get install -y gcc
RUN git clone https://github.com/anonsub29/joern.git

WORKDIR joern


RUN ../gradle-2.0/bin/gradle build -x test



VOLUME /project
WORKDIR ../


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install jupyterlab
RUN pip install wheel
RUN pip install ray
COPY /project/requirements.txt ./
RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


