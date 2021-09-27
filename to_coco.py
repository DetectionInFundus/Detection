# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:07:35 2021

@author: 32068
"""
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import json

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'ma':1,'ex':2,'he':3,'se':0}

def from_divided_generate_names_xml_files(root_dir):
    img_dir = root_dir + '/images'
    names_dir = root_dir + '/namesfiles'
    if not os.path.exists(names_dir):
        os.mkdir(names_dir)

    train_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/train')]
    valid_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/valid')]
    test_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/test')]

    train_namesfile = open(root_dir + '/namesfiles/train.txt', 'w+')
    valid_namesfile = open(root_dir + '/namesfiles/valid.txt', 'w+')
    test_namesfile = open(root_dir + '/namesfiles/test.txt', 'w+')

    train_namesfile.writelines('\n'.join(train_nameslist))
    valid_namesfile.writelines('\n'.join(valid_nameslist))
    test_namesfile.writelines('\n'.join(test_nameslist))

    train_namesfile.close()
    valid_namesfile.close()
    test_namesfile.close()

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename(filename):
    filename = os.path.splitext(filename)[0]
    return filename



def convert(set_type):
    list_fp = open('./namesfiles/'+set_type+'.txt', 'r')# xml_list should be the file of  names of xml
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        xml_f ='./annotations/'+set_type+'/'+line+'.xml'
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = get_filename(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(set_type+'_coco'+'.json', 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()

if __name__ == '__main__':
    from_divided_generate_names_xml_files('.')
    convert("train")
    convert("test")
    convert("valid")