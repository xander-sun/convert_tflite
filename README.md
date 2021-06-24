# convert_tflite

This projetct includes:
1. saved_model to tflite
2. frozen model to tflite
3. checkpoint to tflite


Show signature_def:
1. saved_model
Graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_file_path)   
print(Graph.signature_def)

2. frozen model

