##########################################################################
# File Name: convert_lite.sh
# Author: xander-sun
# mail: youngsun_007@163.com
# Created Time: 2020年11月26日 星期四 18时39分18秒
#########################################################################
#!/bin/sh
#PATH=/home/ys/bin:/usr/local/sbin:/usr/local/bin
#export PATH

tflite_convert \
--input_shape=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
--allow_custom_ops \
--saved_model_dir=/home/ys/tensorflow_test/nets/sycu_ssd_mobilenet_v1_fpn_keras_fine_tune/frozen/saved_model/ \
--output_file=/home/ys/tensorflow_test/nets/sycu_ssd_mobilenet_v1_fpn_keras_fine_tune/tflite/detect.tflite
