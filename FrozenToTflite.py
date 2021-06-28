import tensorflow as tf
#model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/frozen_inference_graph.pb'
#out_model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v3_small_coco_2020_01_14/'
model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb'
out_model_dir = '/home/xs/tensorflow_test/nets/ssd_mobilenet_v1_coco_2018_01_28/'

input_arrays = ['normalized_input_image_tensor'] ### mobilnet_v3 input
#input_arrays = ['image_tensor']      ### mobilnet_v2 input
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']   ### mobilnet_v3 output
#output_arrays = ['detection_boxes','detection_classes','detection_scores','num_detections'] ### mobilnet_v2 output

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(model_dir,
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        input_shapes={input_arrays[0]:[1,300,300,3]}) # path to the SavedModel directory

converter.quantized_input_stats = {input_arrays[0]: (128, 128)}

def to_float32():
    converter.allow_custom_ops=True
    tflite_model = converter.convert()

    # Save the model.
    with open(out_model_dir+'float32_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return "float32"

def to_float16():
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.allow_custom_ops=True
    tflite_model = converter.convert()

    # Save the model.
    with open(out_model_dir+'float16_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return "float16"

def to_uint8():
    #converter.experimental_new_converter = True
    #converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops=True
    #converter.inference_input_type = tf.uint8
    #converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    
    #save the model.
    with open(out_model_dir+'uint8_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return "uint8"
"""
#converter.experimental_new_converter = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#def representative_dataset_gen():
#  for _ in range(num_calibration_steps):
#    # Get sample input data as a numpy array in a method of your choosing.
#    yield [input]
#converter.representative_dataset = representative_dataset_gen

#converter.target_spec.supported_types = [tf.uint8]
#converter.target_spec.supported_types = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8
converter.quantized_input_stats = {input_arrays[0]: (128, 128)}
converter.allow_custom_ops=True
#converter.post_training_quantize=True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the model.
with open(out_model_dir+'float32_model.tflite', 'wb') as f:
#with open(out_model_dir+'float16_model.tflite', 'wb') as f:
    f.write(tflite_model)
"""
if __name__=='__main__':
    model_name = to_float32()
    #model_name = to_float16()
    #model_name = to_uint8()
    print(model_name, "converter success!!!")




