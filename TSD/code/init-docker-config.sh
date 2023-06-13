#!/bin/bash

echo "docker image ID: $1";
echo "docker image name: $2";

docker stop $2 && docker rm $2 && docker run -itd --name $2 $1 /bin/bash
docker exec $2 apt-get update
docker exec $2 apt-get install -y openssh-server vim libgl1-mesa-glx

docker exec $2 pip3 install captum scikit-learn einops pyedflib mne pandas seaborn plotly nni openpyxl opencv-python

docker exec $2 sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
docker exec $2 ssh-keygen -f /root/.ssh/id_ras -N ‘’
docker exec $2 sh -c 'echo "root:root"|chpasswd'

docker stop $2
docker commit $2 mu/pytorch:$2

docker stop $2 && docker rm $2 && docker run -itd --gpus all -p 4884:22 -p 4888:8888 --name $2 -v /home/michael/workspace:/home/michael/workspace -v /mnt:/mnt mu/pytorch:$2 /bin/bash && docker exec $2 service ssh start && docker exec -it $2 /bin/bash

