from PIL import Image, ImageDraw, ImageFont,ImageEnhance
from PIL import Image as im
import random
from matplotlib.pyplot import imshow
import numpy as np
import os
from PIL import ImageEnhance
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import configparser
import json
import ast
import os
from PIL import Image as im
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
from auto_training import DataGenerator
from scaling import DataScaling
from generate_tfrecord import GenerateTf
from pipeline_update import ObjectDetectionPipeline
from training import TrainData
import shutil
from shutil import copyfile
config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)

raw_image_path = config.get('scale_config_raw','raw_image_path')
raw_xml_path = config.get('scale_config_raw','raw_xml_path')
modify_img_raw = config.get('scale_config_raw', 'modify_img_path')
modify_xml_raw = config.get('scale_config_raw', 'modify_xml_path')


if not os.path.exists(modify_img_raw):
       os.makedirs(modify_img_raw)
if not os.path.exists(modify_xml_raw):
       os.makedirs(modify_xml_raw)

final_images = config.get('final_data', 'final_images')
final_xmls = config.get('final_data', 'final_xmls')
if not os.path.exists(final_images):
       os.makedirs(final_images)
if not os.path.exists(final_xmls):
       os.makedirs(final_xmls)





final_images = config.get('final_data', 'final_images')
final_xmls = config.get('final_data', 'final_xmls')
generate_images = DataGenerator()
generate_scaled= DataScaling()
generate_tf_records = GenerateTf()
update_pipeline = ObjectDetectionPipeline()
train=TrainData()

def move_data(path1,path2):
    for file in os.listdir(path1):
        shutil.copy(os.path.join(path1,file), path2)





def final_execution():

    if FLAG=="include":
       functions_include_raw = [
             generate_scaled.generate_scaled(raw_image_path, raw_xml_path, modify_img_raw,modify_xml_raw)
             move_data(raw_image_path,final_images),move_data(raw_xml_path,final_xmls),
             move_data(modify_img_raw,final_images),move_data(modify_xml_raw,final_xmls),
             generate_tf_records.main(),update_pipeline.save_pipeline_config(),train.train(),train.model_graph()]
       for func in functions_include_raw:
           try:
               func
           except ValueError:
               break
   


if __name__ == "__main__":
    final_execution()
