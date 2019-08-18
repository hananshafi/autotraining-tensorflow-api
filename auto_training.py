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
config = configparser.RawConfigParser()
configFilePath = r'config.txt'
config.read(configFilePath)
print('generating data.............')
class DataGenerator:
    noise_paths =[]
    words =[]
    bgs =[]
    font_styles=[]
    template_list =[]
    colors = ast.literal_eval(config.get('aug_config', 'colors'))
    augmentaion_modes = ast.literal_eval(config.get('aug_config', 'augmentaion_modes'))
    scale_modes = ast.literal_eval(config.get('aug_config', 'scale_modes'))
    scale_vals = ast.literal_eval(config.get('aug_config', 'scale_vals'))
    augment_factor = json.loads(config.get('aug_config', 'augment_factor'))
    no_examples = config.getint('aug_config', 'no_examples')
    def __init__(self):
        self.image_path = config.get('aug_config', 'image_path')
        self.save_image_path = config.get('aug_config', 'save_image_path')
        self.save_xml_path = config.get('aug_config', 'save_xml_path')
        self.noise_path =config.get('aug_config', 'noise_path')
        self.word_path = config.get('aug_config', 'word_path')
        self.background_path =config.get('aug_config', 'background_path')
        self.font_path = config.get('aug_config', 'font_path')
        
        if not os.path.exists(self.save_image_path):
                os.makedirs(self.save_image_path)
        if not os.path.exists(self.save_xml_path):
                os.makedirs(self.save_xml_path)

        
        for root, templates, files in os.walk(self.image_path):
            for template in templates:
                self.template_list.append(os.path.join(self.image_path,template))
                #for r1, f1, files in os.walk(os.path.join(self.image_path, template)):
                    #for file in files:
                        #t_path = os.path.join(self.image_path, template, file)
                        #self.template_paths.append(t_path)

        for root, templates, files in os.walk(self.noise_path):
            for file in files:
                n_path = os.path.join(self.noise_path, file)
                self.noise_paths.append(n_path)

        with open(self.word_path, 'rb') as f:
            lines = f.readlines()
            for line in lines:
                self.words.append(line.strip().decode('UTF-8'))

        for root, folder, files in os.walk(self.background_path):
            for f in files:
                self.bgs.append(f)

        for root, folder, files in os.walk(self.font_path):
            for f in files:
                self.font_styles.append(f)


    def augment_image(self, image, mode, factor):
        if mode == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
        if mode == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
        if mode == 'color':
            enhancer = ImageEnhance.Color(image)
        if mode == 'sharpness':
            enhancer = ImageEnhance.Sharpness(image)
        adjusted_image = enhancer.enhance(factor)
        return adjusted_image

    def generate_images(self):
        cnt = 1
        for i in range(self.no_examples):
            bg = random.choice(self.bgs)
            image = Image.open('back_ground/' + bg).resize((1024, 768))
            draw = ImageDraw.Draw(image)
            w, h = image.size
            f_name = bg.strip().split(".")[0]

            # label xml generation
            root = Element('annotation')
            folder = SubElement(root, 'folder')
            folder.text = "aa_xml"

            filename = SubElement(root, 'filename')
            filename.text = str(cnt) + ".jpg"

            path = SubElement(root, 'path')
            path.text = "path"

            source = SubElement(root, 'source')
            database = SubElement(source, 'database')
            database.text = 'unknown'

            size = SubElement(root, 'size')
            width = SubElement(size, 'width')
            width.text = str(w)
            height = SubElement(size, 'height')
            height.text = str(h)
            depth = SubElement(size, 'depth')
            depth.text = str(3)

            segmented = SubElement(root, 'segmented')
            segmented.text = '0'
            pos_w = 20
            pos_h = 20

            while pos_h < h - 10:
                max_height = 0
                pos_w = 10
                choose_line = random.choice([True, False])
                dm_height = random.choice(range(5, 55))
                if choose_line:

                    while pos_w < w:
                        choose_word = random.choice([True, False])
                        if not choose_word:
                            choose_text = random.choice([True, False])
                        if choose_word:
                            choose_noise = random.choice([False, False, False, True])
                            if not choose_noise:
                                template_paths=[]
                                template = random.choice(self.template_list)
                                #print(template)
                                for r1, f1, files in os.walk(os.path.join(template)):
                                    for file in files:
                                        t_path = os.path.join(template, file)
                                        template_paths.append(t_path)
                                template_path = random.choice(template_paths)
                                template_name = template_path.strip().split("/")[1]
                                template = im.open(template_path)
                                mode = random.choice(self.augmentaion_modes)
                                #mode = "same"
                                scale_mode = random.choice(self.scale_modes)
                                scale_val = random.choice(self.scale_vals)
                                if scale_mode == 1:
                                    t_w, t_h = template.size
                                    mod_w, mod_h = int(t_w * scale_val), int(t_h * scale_val)
                                    template = template.resize((mod_w, mod_h))
                                if not mode == 'same':
                                    factor = random.choice(self.augment_factor[mode])
                                    template = self.augment_image(template, mode, factor)
                                t_size_w, t_size_h = template.size
                                if t_size_h > max_height:
                                    max_height = t_size_h
                                if (pos_w + t_size_w) < w - 10 and (pos_h + t_size_h) < h - 10:
                                    (x, y) = (pos_w, pos_h)
                                    image.paste(template, (x, y))
                                    xmax_c = x + t_size_w
                                    ymax_c = y + t_size_h

                                    # add object in xml
                                    obj = SubElement(root, 'object')
                                    name = SubElement(obj, 'name')
                                    name.text = template_name
                                    pose = SubElement(obj, 'pose')
                                    pose.text = 'Unspecified'
                                    truncated = SubElement(obj, 'truncated')
                                    truncated.text = '0'
                                    difficult = SubElement(obj, 'difficult')
                                    difficult.text = '0'

                                    bndbox = SubElement(obj, 'bndbox')
                                    xmin = SubElement(bndbox, 'xmin')
                                    xmin.text = str(x)
                                    ymin = SubElement(bndbox, 'ymin')
                                    ymin.text = str(y)
                                    xmax = SubElement(bndbox, 'xmax')
                                    xmax.text = str(xmax_c)
                                    ymax = SubElement(bndbox, 'ymax')
                                    ymax.text = str(ymax_c)

                                    pos_w = pos_w + t_size_w + 10

                                else:
                                    pos_h = pos_h + max_height + 10
                                    break
                            else:
                                # Logic to add noise icon without creating labels
                                template_path = random.choice(self.noise_paths)
                                template_name = template_path.strip().split("/")[1]
                                template = im.open(template_path)
                                mode = random.choice(self.augmentaion_modes)
                                scale_mode = random.choice(self.scale_modes)
                                scale_val = random.choice(self.scale_vals)
                                if scale_mode == 1:
                                    t_w, t_h = template.size
                                    mod_w, mod_h = int(t_w * scale_val), int(t_h * scale_val)
                                    template = template.resize((mod_w, mod_h))
                                if not mode == 'same':
                                    factor = random.choice(self.augment_factor[mode])
                                    template = self.augment_image(template, mode, factor)
                                t_size_w, t_size_h = template.size
                                if t_size_h > max_height:
                                    max_height = t_size_h
                                if (pos_w + t_size_w) < w - 10 and (pos_h + t_size_h) < h - 10:
                                    (x, y) = (pos_w, pos_h)
                                    image.paste(template, (x, y))
                                    xmax_c = x + t_size_w
                                    ymax_c = y + t_size_h

                                    pos_w = pos_w + t_size_w + 10

                                else:
                                    pos_h = pos_h + max_height + 10
                                    break
                        elif choose_text:

                            word = random.choice(self.words)
                            font_style = random.choice(self.font_styles)
                            font_size = random.choice(range(10, 55))
                            font = ImageFont.truetype("fonts/" + font_style, font_size)
                            word_size_w, word_size_h = font.getsize(str(word))
                            if word_size_h > max_height:
                                max_height = word_size_h
                            font_color = random.choice(self.colors)
                            if (pos_w + word_size_w) < w - 10 and (pos_h + word_size_h) < h - 10:
                                (x, y) = (pos_w, pos_h)
                                draw.text((x, y), word, fill=font_color, font=font)
                                pos_w = pos_w + word_size_w + 10
                            else:
                                pos_h = pos_h + max_height + 10
                                break
                        else:
                            pos_w = pos_w + 10
                    pos_h = pos_h + max_height + 10

                else:
                    pos_h = pos_h + dm_height + 10
            image.save(os.path.join(self.save_image_path, str(cnt) + '.jpg'))
            tree = ET.ElementTree(root)
            tree.write(os.path.join(self.save_xml_path, str(cnt) + ".xml"))
            cnt += 1
