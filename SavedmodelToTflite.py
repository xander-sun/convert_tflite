import tensorflow as tf

model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/frozen_inference_graph.pb'
out_model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/'

input_arrays = ['normalized_input_image_tensor'] ### mobilnet_v3 input
#input_arrays = ['image_tensor']      ### mobilnet_v2 input
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',
'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']   ### mobilnet_v3 output
#output_arrays = ['detection_boxes','detection_classes','detection_scores','num_detections'] ### mobilnet_v2 output

#tf.enable_control_flow_v2()

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_dir) # path to the SavedModel direct
concrete_func = converter.signatures['serving_default']
# Set the shape of the input in the concrete function.
concrete_func.inputs[0].set_shape([1,320,320,3]) # I also tried with [1,300,300,3]

# Convert the model to a TFLite model.
converter =  tf.compat.v1.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the model.
with open(out_model_dir+'INT8_model.tflite', 'wb') as f:
    f.write(tflite_model)
