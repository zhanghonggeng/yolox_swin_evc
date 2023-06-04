import os
from PIL import Image

root_dir = "./dataset/DOTA_V1.0/"
#============================================
# 采用水平框hbb 训练、
#============================================
# 生成train的xml
# annotations_dir = root_dir + "train/labelTxt-v1.0/trainset_reclabelTxt/"
# image_dir = root_dir + "train/images/"
# xml_dir = root_dir + "train/labelTxt-v1.0/xml/"

class_name = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                   'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',
                   'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
                   ]

# 生成val的xml
annotations_dir = root_dir + "val/labelTxt-v1.0/valset_reclabelTxt/"
image_dir = root_dir + "val/images/"
xml_dir = root_dir + "val/labelTxt-v1.0/xml/"

def txt_xml(filename):
    file_path = annotations_dir+filename
    fin = open(file_path, 'r')
    fin_lines = fin.readlines()
    image_name = filename.split('.')[0]

    img = Image.open(image_dir + image_name + ".png")
    xml_name = xml_dir + image_name + '.xml'
    with open(xml_name, 'w') as fout:
        fout.write('<annotation>' + '\n')

        fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
        fout.write('\t' + '<filename>' + image_name + '.png' + '</filename>' + '\n')

        fout.write('\t' + '<source>' + '\n')
        fout.write('\t\t' + '<database>' + 'DOTAV1.0' + '</database>' + '\n')
        fout.write('\t' + '</source>' + '\n')

        fout.write('\t' + '<size>' + '\n')
        fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
        fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
        fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
        fout.write('\t' + '</size>' + '\n')

        fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

        for line in fin_lines:
            a = '\t' in line
            if a:
                line_content = line.replace('\t',',').replace('\n','').split(',')
            else:
                line_content = line.replace(' ', ',').replace('\n', '').split(',')
            fout.write('\t' + '<object>' + '\n')
            fout.write('\t\t' + '<name>' + line_content[-2] + '</name>' + '\n')  # true
            fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
            fout.write('\t\t' + '<truncated>' + '0' + '</truncated>' + '\n')
            fout.write('\t\t' + '<difficult>' + line_content[-1] + '</difficult>' + '\n')  # true
            fout.write('\t\t' + '<bndbox>' + '\n')
            fout.write('\t\t\t' + '<xmin>' + line_content[0] + '</xmin>' + '\n')  # true
            fout.write('\t\t\t' + '<ymin>' + line_content[1] + '</ymin>' + '\n')  # true
            # pay attention to this point!(0-based)
            fout.write('\t\t\t' + '<xmax>' + line_content[4] + '</xmax>' + '\n')  # true
            fout.write('\t\t\t' + '<ymax>' + line_content[5] + '</ymax>' + '\n')  # true
            fout.write('\t\t' + '</bndbox>' + '\n')
            fout.write('\t' + '</object>' + '\n')

        fin.close()
        fout.write('</annotation>')


for file_name in os.listdir(annotations_dir):
    txt_xml(file_name)


