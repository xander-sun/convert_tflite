##########################################################################
# File Name: get_frozen_pb.sh
# Author: xander-sun
# mail: youngsun_007@163.com
# Created Time: 2020年11月26日 星期四 15时55分31秒
#########################################################################
#!/bin/sh
#PATH=/home/ys/bin:/usr/local/sbin:/usr/local/bin
#export PATH

train_code_path="/home/xs/tensorflow_test/models/research/object_detection"
pb_path="/home/xs/tensorflow_test/nets/ssd_mobilenet_v1_coco_2018_01_28"
python3 ${train_code_path}/export_tflite_ssd_graph.py \
--pipeline_config_path=${pb_path}/pipeline.config \
--trained_checkpoint_prefix=${pb_path}/model.ckpt \
--output_directory=${pb_path}

