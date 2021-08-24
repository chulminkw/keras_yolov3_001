import glob
import os
import xml.etree.ElementTree as ET

IMAGE_DIR = r'C:\Users\q\raccoon\images'

def xml_to_csv(path, output_filename):
    xml_list = []
    # xml 확장자를 가진 모든 파일의 절대 경로로 xml_file할당.
    with open(output_filename, "w") as train_csv_file:
        for xml_file in glob.glob(path + '/*.xml'):
            # xml 파일을 parsing하여 XML Element형태의 Element Tree를 생성하여 object 정보를 추출.
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # 파일내에 있는 모든 object Element를 찾음.
            full_image_name = os.path.join(IMAGE_DIR, root.find('filename').text)
            value_str_list = ' '
            for obj in root.findall('object'):
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                # 단 하나의 class_id raccoon
                class_id = 0
                value_str = ('{0},{1},{2},{3},{4}').format(x1, y1, x2, y2, class_id)
                value_str_list = value_str_list + value_str + ' '
                print('value_str_list:', value_str_list, full_image_name)
                # object별 정보를 tuple형태로 object_list에 저장.
            train_csv_file.write(full_image_name + ' ' + value_str_list + '\n')
        # xml file 찾는 for loop 종료

xml_to_csv(r'C:\Users\q\raccoon\annotations', r'C:\Users\q\raccoon\annotations\raccoon_anno.csv')