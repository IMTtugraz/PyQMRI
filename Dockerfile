#Download base image python buster
FROM nvidia/opencl

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive 
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update

RUN apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-tk && \
    apt-get install -y ocl-icd* opencl-headers &&\
    apt-get install -y libclfft* &&\
    apt-get install -y git

ENV JENKINS_HOME /var/jenkins_home
ENV JENKINS_SLAVE_AGENT_PORT 50000
RUN useradd -d "$JENKINS_HOME" -u 971 -m -s /bin/bash jenkins
VOLUME /var/jenkins_home

RUN ls -lah /var/jenkins_home

ENV PATH="/var/jenkins_home/.local:${PATH}"

RUN pip3 install cython 
RUN pip3 install pyopencl
RUN git clone https://github.com/geggo/gpyfft.git &&\
    pip3 install gpyfft/. &&\
    pip3 install pytest &&\
    pip3 install pytest-cov &&\
    pip3 install pylint &&\
    pip3 install pylint_junit

