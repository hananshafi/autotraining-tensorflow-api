This is an automatic training mechanism for Tensorflow Object detection API.
All the changes and modifications are to be done in Config file. The config file has some Optional Parameters and some mandatory parameters.
Download the API here: https://github.com/tensorflow/models/tree/master/research/object_detection

Put all the these files in the tensorflow API folder.



[scale_config_raw]
#generating augmented scaled data from raw data
   Optional Parameters:
         - scale_vals
   
   Mandatory parameteres:
         - FLAG: it specifies whether raw data is to be included in the training.
         - raw_image_path : path to raw images
         - raw_xml_path: path to raw xmls
         - modify_img_path: path for storing scaled raw images
         - modify_xml_path: path for storing the corresponding scaled raw xmls
         - num_varaiations_per_image: Number of variations to be generated per image

[final_data]
contains path to final data (synthetic + scaled)
         - final_images:path to final image folder
         - final_xmls: path to final xml folder

[tf_records]
contains path to tf_records and label map
         - output_path = path to training records
         - label_map_path = path to label map

[training_parameters]
contains parameters for training
train_dir/checkpoint dir and saved_model dir will be created automatically.No need to create them manually.
         - num_steps: Number of training steps
         - pipeline_path: path to pipeline config file (#after modification this will get written)
         - train_dir: folder for storing training progress/ checkpoints
         - checkpoint_dir: same as training dir
         - saved_model: path for inference graph
