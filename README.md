# convert_tflite

Config needed:
1. tensorflow/models
git@github.com:tensorflow/models.git
2. model trained by tensorflow1.15
3. tensorflow1.15


This projetct includes:
1. saved_model to tflite
2. tflitefrozen model to tflite    (finished)
3. checkpoint to tflite             (finished)

From checkpoint to tflite:
1. sh CovertTFliteFrozenFromCheckpoint.sh
   get two files: tflite_graph.pb, tflite_graph.pbtxt
   you can get node names in tflite_graph.pbtxt
2. python3 FrozenToTflite.py
   get float16_model.tflite or uint8_model.tflite

Show signature_def:
1. saved_model
Graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_file_path)   
print(Graph.signature_def)

2. frozen model
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        ####graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
print(graph_def)

3. checkpoint
saver=tf.train.import_meta_graph('./ssd_mobilenet_v1_sycu_20210328/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./ssd_mobilenet_v1_sycu_20210328/'))
####saver.restore(sess=sess, save_path=args.save_path)  # 读取保存的模型
detect_graph = tf.get_default_graph()
print(detect_graph.signature_def)

