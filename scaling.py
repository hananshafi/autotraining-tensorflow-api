import os
from PIL import Image as im
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import configparser
import ast
config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)

class DataScaling:
    def __init__(self):
        self.scale_vals = ast.literal_eval(config.get('scale_config', 'scale_vals'))
        """self.image_folder = config.get('aug_config', 'save_image_path')
        self.xml_folder = config.get('aug_config', 'save_xml_path')
        self.modify_img_folder = config.get('scale_config', 'modify_img_folder')
        self.modify_xml_folder = config.get('scale_config', 'modify_xml_folder')
        self.raw_image_path = config.get('scale_config_raw','raw_image_path')
        self.raw_xml_path = config.get('scale_config_raw','raw_xml_path')
        self.modify_img_raw = config.get('scale_config_raw', 'modify_img_folder')
        self.modify_xml_raw = config.get('scale_config_raw', 'modify_xml_folder')"""
        self.num_variations_per_image = config.getint('scale_config', 'num_varaiations_per_image')
        

    def generate_scaled(self,image_folder,xml_folder,modify_image_path,modify_xml_path):
        print('scaling_data.........')
        for root, directory, files in os.walk(image_folder):
            for image_file in files:
                file_name, ext = image_file.strip().split(".")
                if not ext == "jpg":
                    continue

                screen = im.open(image_folder + "/" + image_file)
                width, height = screen.size
                scale = np.random.choice(self.scale_vals)
                new_width, new_height = int(width * scale), int(height * scale)
                new_screen = screen.resize((new_width, new_height))

                for i in range(self.num_variations_per_image):
                    screen = im.open(image_folder + "/" + image_file)
                    width, height = screen.size
                    scale = np.random.choice(self.scale_vals)
                    new_width, new_height = int(width * scale), int(height * scale)
                    new_screen = screen.resize((new_width, new_height))
                    small_distort_flag = np.random.choice([True, False])
                    if small_distort_flag:
                        max_width = new_width + 31
                        max_height = new_height + 31
                    else:
                        max_width = 2 * new_width
                        max_height = 2 * new_height
                    bg_width = np.random.choice(range(new_width + 1, max_width))
                    bg_height = np.random.choice(range(new_height + 1, max_height))
                    bg_size = (bg_width, bg_height)
                    border_x = bg_width - new_width
                    border_y = bg_height - new_height

                    paste_x = np.random.choice(range(border_x))
                    paste_y = np.random.choice(range(border_y))

                    bg = im.new('RGB', bg_size)
                    bg.paste(new_screen, (paste_x, paste_y))

                    xml_path = xml_folder + "/" + file_name + ".xml"
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for item in root:
                        if item.tag == 'filename':
                            item.text = file_name + "_" + str(i) + ".jpg"
                        elif item.tag == 'object':
                            for ele in item:
                                if ele.tag == 'bndbox':
                                    for coord in ele:
                                        val = int(int(coord.text) * scale)
                                        if coord.tag == 'xmin' or coord.tag == 'xmax':
                                            val = val + paste_x
                                        else:
                                            val = val + paste_y

                                        coord.text = str(val)

                    bg.save(modify_image_path + "/" + file_name + "_" + str(i) + ".jpg")
                    tree2 = ET.ElementTree(root)
                    tree2.write(modify_xml_path + "/" + file_name + "_" + str(i) + ".xml")
