
import configparser
import os
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from tensorflow.python.lib.io import file_io
import configparser
#from generate_tfrecord import GenerateTf
import xml_to_csv
pipeline_config_path = r'pipeline.config'

config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)

#labels= GenerateTf()

class ObjectDetectionPipeline:
    def __init__(self):
        self.xml_path=config.get('tf_records', 'xml_path')
        self.num_steps = config.getint('training_parameters', 'num_steps')
        self.label_map = config.get('tf_records', 'label_map_path')
        self.train_input = config.get('tf_records', 'output_path')
        self.eval_input = config.get('tf_records', 'output_path')
        self.pipeline_config_path= config.get('training_parameters', 'pipeline_path')
        #self.num_classes = labels.num_classes()
        #self.save_pipeline_directory= config.get('training_parameters', 'save_pipeline_directory')


    def num_classes(self):
            xml_path = os.path.join(os.getcwd(), self.xml_path)
            examples = xml_to_csv.xml_to_csv(xml_path)
            # examples = pd.read_csv(FLAGS.csv_input)
            label_classes = set(examples['class'].tolist())
            label2int = {}
            for val, label in enumerate(label_classes):
                label2int[label] = val + 1
            labels = []
            for keys,values in label2int.items():
                labels.append(values)
            return max(labels)
    def get_configs_from_pipeline_file(self,config_override=None):

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(self.pipeline_config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)
        if config_override:
            text_format.Merge(config_override, pipeline_config)
        # print(pipeline_config)
        return pipeline_config

    def create_configs_from_pipeline_proto(self):
        pipeline_config = self.get_configs_from_pipeline_file(config_override=None)
        configs = {}
        configs['model']= pipeline_config.model
        configs["num_classes"] = pipeline_config.model.faster_rcnn.num_classes
        configs["num_steps"] = pipeline_config.train_config.num_steps
        configs["train_config"] = pipeline_config.train_config
        configs["train_input_config"] = pipeline_config.train_input_reader
        configs["train_label_path"] = pipeline_config.train_input_reader.label_map_path
        configs["train_input_path"] = pipeline_config.train_input_reader.tf_record_input_reader.input_path
        configs["eval_config"] = pipeline_config.eval_config
        configs["eval_input_configs"] = pipeline_config.eval_input_reader
        configs["eval_input_path"] = pipeline_config.eval_input_reader.tf_record_input_reader.input_path
        configs["eval_label_path"] = pipeline_config.eval_input_reader.label_map_path
        # Keeps eval_input_config only for backwards compatibility. All clients should
        # read eval_input_configs instead.
        if configs["eval_input_configs"]:
            configs["eval_input_config"] = configs["eval_input_configs"]
        if pipeline_config.HasField("graph_rewriter"):
            configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

        return configs

    def create_pipeline_proto_from_configs(self):
        configs = self.create_configs_from_pipeline_proto()
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        pipeline_config.model.CopyFrom(configs["model"])
        pipeline_config.model.faster_rcnn.num_classes=self.num_classes()
        pipeline_config.train_config.CopyFrom(configs["train_config"])
        pipeline_config.train_config.num_steps= self.num_steps
        pipeline_config.train_input_reader.CopyFrom(configs["train_input_config"])
        pipeline_config.train_input_reader.label_map_path=self.label_map
        del pipeline_config.train_input_reader.tf_record_input_reader.input_path[:]
        pipeline_config.train_input_reader.tf_record_input_reader.input_path.append(self.train_input)
        pipeline_config.eval_config.CopyFrom(configs["eval_config"])
        pipeline_config.eval_input_reader.CopyFrom(configs["eval_input_configs"])
        pipeline_config.eval_input_reader.label_map_path=self.label_map
        del pipeline_config.eval_input_reader.tf_record_input_reader.input_path[:]
        pipeline_config.eval_input_reader.tf_record_input_reader.input_path.append(self.eval_input)
        if "graph_rewriter_config" in configs:
            pipeline_config.graph_rewriter.CopyFrom(configs["graph_rewriter_config"])
        return pipeline_config

    def save_pipeline_config(self):
        print("updating pipeline..........")
        pipeline_config = self.create_pipeline_proto_from_configs()
        directory = os.getcwd()
        if not file_io.file_exists(directory):
            file_io.recursive_create_dir(directory)
        pipeline_config_path = os.path.join(directory, "pipeline.config")
        config_text = text_format.MessageToString(pipeline_config)
        with tf.gfile.Open(pipeline_config_path, "wb") as f:
            tf.logging.info("Writing pipeline config file to %s",
                            pipeline_config_path)
            f.write(config_text)
