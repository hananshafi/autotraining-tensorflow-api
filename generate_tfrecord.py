"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf


from PIL import Image
import dataset_util
from collections import namedtuple, OrderedDict
import pickle

import xml_to_csv

'''flags = tf.app.flags
flags.DEFINE_string('xml_path', '', 'Path to xmls')
flags.DEFINE_string('image_path', '', 'Path to xmls')
flags.DEFINE_string('label_map_path', '', 'Path to xmls')
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS'''


import configparser
#pickle_in = open('object_detection/projection_screen/data/label.pkl','rb')
#label2int = pickle.load(pickle_in)

config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)

class GenerateTf:
    def __init__(self):
        self.image_path = config.get('tf_records', 'image_path')
        self.xml_path = config.get('tf_records', 'xml_path')
        self.output_path = config.get('tf_records', 'output_path')
        self.label_map_path = config.get('tf_records', 'label_map_path')


# TO-DO replace this with label map
    def class_text_to_int(self,row_label):
        print(row_label)
        if row_label in label2int:
            return label2int[row_label]
        else:
            return None


    def split(self,df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(self,group, path,label2int):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(label2int[row['class']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


    def main(self):
        print('generating tf records........')
        writer = tf.python_io.TFRecordWriter(self.output_path)
        path = os.path.join(os.getcwd(), self.image_path)

        xml_path = os.path.join(os.getcwd(), self.xml_path)
        examples = xml_to_csv.xml_to_csv(xml_path)
        #examples = pd.read_csv(FLAGS.csv_input)
        label_classes = set(examples['class'].tolist())
        label2int={}
        for val,label in enumerate(label_classes):
            label2int[label]=val+1
        #generate label map
        label_map_path = self.label_map_path
        with open(label_map_path,'w') as f:
            labels=[]
            for label in label2int:
                f.write("item {\n")
                f.write("  id: " + str(label2int[label]) + '\n')
                f.write("  name: \'" + label + "\'\n")
                f.write("}\n\n")
                labels.append(label2int[label])

        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, path,label2int)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), self.output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))

    """def num_classes(self):
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
            return max(labels)"""
