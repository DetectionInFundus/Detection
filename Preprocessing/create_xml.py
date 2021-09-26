import numpy as np
import cv2
from skimage import measure
import xml.etree.ElementTree as ET

# for IDRiD A.Segmentation


def getBbxAndNum(rgb_image):
    mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    mask[np.where(np.all(rgb_image == [255, 255, 255], axis=-1))[:2]] = 1
    label_image = measure.label(mask)
    bbx_num = len(measure.regionprops(label_image))
    bbxs = []
    for region in measure.regionprops(label_image):
        bbxs.append(region.bbox)
    return bbxs, bbx_num


def createXML_1(foldername='IDRiD', filename='unknown.jpg', size=(600, 600, 3)):
    # <annotation></annotation>
    root = ET.Element("annotation")
    # <folder></folder>
    folder_node = ET.Element("folder")
    folder_node.text = foldername
    # <filename></filename>
    file_node = ET.Element("filename")
    file_node.text = filename
    # <size></size>
    size_node = ET.Element("size")
    width_node = ET.SubElement(size_node, "width")
    height_node = ET.SubElement(size_node, "height")
    depth_node = ET.SubElement(size_node, "depth")
    width_node.text = str(size[0])
    height_node.text = str(size[1])
    depth_node.text = str(size[2])
    # append together
    root.append(folder_node)
    root.append(file_node)
    root.append(size_node)
    return root


def createXML_2(root, bbxs, bbx_num, classname):
    # <object></object>
    for i in range(bbx_num):
        object_node = ET.Element("object")
        # <name></name>
        name_node = ET.Element("name")
        name_node.text = classname
        # <pose></pose>
        pose_node = ET.Element("pose")
        pose_node.text = 'Unspecified'
        # <truncated></truncated>
        trunc_node = ET.Element("truncated")
        trunc_node.text = '0'
        # <difficult></difficult>
        dif_node = ET.Element("difficult")
        dif_node.text = '0'
        # <bndbox></bndbox>
        bbx_node = ET.Element("bndbox")
        # bbx information
        xmin_node = ET.SubElement(bbx_node, "xmin")
        ymin_node = ET.SubElement(bbx_node, "ymin")
        xmax_node = ET.SubElement(bbx_node, "xmax")
        ymax_node = ET.SubElement(bbx_node, "ymax")
        xmin_node.text = str(bbxs[i][1])
        ymin_node.text = str(bbxs[i][0])
        xmax_node.text = str(bbxs[i][3])
        ymax_node.text = str(bbxs[i][2])

        # append together
        object_node.append(name_node)
        object_node.append(pose_node)
        object_node.append(trunc_node)
        object_node.append(dif_node)
        object_node.append(bbx_node)
        root.append(object_node)
    return root


def createXML_3(root, filename, target_dir):
    filename = filename.split('.')[0]
    tree = ET.ElementTree(root)
    print('storepath: ' + target_dir + '/' + filename + '.xml')
    tree.write(target_dir + '/' + filename + '.xml')

