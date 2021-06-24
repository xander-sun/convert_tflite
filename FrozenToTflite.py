import tensorflow as tf

model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/frozen_inference_graph.pb'
out_model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/'

input_arrays = ['normalized_input_image_tensor'] ### mobilnet_v3 input
#input_arrays = ['image_tensor']      ### mobilnet_v2 input
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',
'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']   ### mobilnet_v3 output
#output_arrays = ['detection_boxes','detection_classes','detection_scores','num_detections'] ### mobilnet_v2 output

#tf.enable_control_flow_v2()
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(model_dir,
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        input_shapes={input_arrays[0]:[1,320,320,3]}) # path to the SavedModel directory
converter.target_spec.supported_types = [tf.float16]
#converter.inference_input_type = tf.float16
#converter.quantized_input_stats = {"normalized_input_image_tensor": (128, 128)} ### mobilnet_v3 convert input
converter.quantized_input_stats = {input_arrays[0]: (128, 128)}
converter.allow_custom_ops=True
converter.post_training_quantize=True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_dir) # path to the SavedModel direct
#concrete_func = converter.signatures['serving_default']
# Set the shape of the input in the concrete function.
#concrete_func.inputs[0].set_shape([1,320,320,3]) # I also tried with [1,300,300,3]

# Convert the model to a TFLite model.
#converter =  tf.compat.v1.TFLiteConverter.from_concrete_functions([concrete_func])
#converter.optimizations =  [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the model.
with open(out_model_dir+'float16_model.tflite', 'wb') as f:
    f.write(tflite_model)
