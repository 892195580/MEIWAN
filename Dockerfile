# step1 设置base image
#FROM hub.data.wust.edu.cn:30880/library/tensorflow:1.14.0-gpu-py3
#RUN pip install --upgrade pip

FROM hub.data.wust.edu.cn:30880/zhu/zhu-env-rel:latest

# step2 安装相应依赖
#WORKDIR /home
#RUN apt-get update \
#&& apt-get install wget curl vim gcc zlib1g-dev bzip2 -y \
#&& apt-get install zlib1g.dev \
# 安装libssl1.0-dev解决pip安装时md5模块无法导入问题
#&& apt-get install openssl libssl1.0-dev -y \
#&& apt-get install g++ build-essential -y \
# && apt-get install python-tk -y \
# && apt-get install tk-dev -y \
#&& mkdir /usr/local/source \

# step3 将工程下面的机器学习相关的文件复制到容器某个目录中，例如：/home/mnist
COPY ./DocuNet-main /home/DocuNet-main

# step4 设置容器中的工作目录，直接切换到/home/mnist目录下
WORKDIR /home/DocuNet-main

# step5 安装依赖
#RUN pip install -r requirements.txt

# step6 设置容器启动时的运行命令，这里我们直接运行python程序
ENTRYPOINT ["python", "/home/DocuNet-main/train_balanceloss.py"]

#&& cd /usr/local/source \
#&& pip install Cython==0.29.7 \
#&& pip install pyyaml==5.1 \
#&& pip install numpy==1.16.3 \
#&& pip install pandas==0.23.2 \
#&& pip install scipy==1.2.1 \
#&& pip install scikit-learn==0.20.3 \
#&& pip install matplotlib==2.1.0 \
#&& pip install lightgbm==2.2.2 \
#&& pip install catboost==0.13.1 \
#&& pip install grpcio==1.20.1 \
#&& pip install grpcio-tools==1.20.1 \