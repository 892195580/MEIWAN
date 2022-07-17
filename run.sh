#! /bin/bash
echo "Zhuandong1994" | docker login hub.data.wust.edu.cn:30880 -u "zhuandong" --password-stdin
docker build -f ./Dockerfile -t hub.data.wust.edu.cn:30880/zhu/zhu-docunet:wn41 .
docker push hub.data.wust.edu.cn:30880/zhu/zhu-docunet:wn41