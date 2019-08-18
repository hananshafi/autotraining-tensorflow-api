import configparser
import os
config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)
train_dir = config.get('training_parameters', 'train_dir')
saved_model = config.get('training_parameters', 'saved_model')
num_steps = config.getint('training_parameters', 'num_steps')
pipeline_path = config.get('training_parameters', 'pipeline_path')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(saved_model):
    os.makedirs(saved_model)

class TrainData:

    def __init__(self):

        self.train_command1 = "export PYTHONPATH=`pwd`:`pwd`/slim; python object_detection/legacy/train.py --logtostderr --train_dir={} --pipeline_config_path={}".format(train_dir,pipeline_path)
        self.train_command2 = "export PYTHONPATH=`pwd`:`pwd`/slim; python object_detection/model_main.py --pipeline_config_path={} --model_dir={} --num_train_steps={} --sample_1_of_n_eval_examples=10 --alsologtostderr".format(pipeline_path, train_dir, num_steps)
        #self.eval = "CUDA_VISIBLE_DEVICES="" python object_detection/legacy/eval.py --logtostderr --pipeline_config_path='{}' --checkpoint_dir='{}' --eval_dir='{}'".format()
        self.generate_graph = "export PYTHONPATH=`pwd`:`pwd`/slim;python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path {} --trained_checkpoint_prefix {}/model.ckpt-{} --output_directory {}".format(pipeline_path,train_dir,num_steps,saved_model)
    def train(self):
        os.system(self.train_command1)

    def model_graph(self):
        os.system(self.generate_graph) 
   