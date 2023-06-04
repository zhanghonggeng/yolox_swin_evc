import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path = 'model_data/DOTA_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_DOTA_path = 'dataset'

VOCdevkit_DOTA_sets = [('2007', 'train'), ('2007', 'val')]


def convert_annotation(year, image_id, list_file,image_set):
    # in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    in_file = open(os.path.join(VOCdevkit_DOTA_path, 'DOTA_V1.0/%s/labelTxt-v1.0/xml/%s.xml' % (image_set, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        # xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        # saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')

        # 生成dota的txt
        train_xmlfilepath = os.path.join(VOCdevkit_DOTA_path,'DOTA_V1.0/train/labelTxt-v1.0/xml')
        val_xmlfilepath = os.path.join(VOCdevkit_DOTA_path,'DOTA_V1.0/val/labelTxt-v1.0/xml')
        test_imgnamepath = os.path.join(VOCdevkit_DOTA_path,'DOTA_V1.0/test/images')
        # xmlfilepath = os.path.join(VOCdevkit_DOTA_path, 'DOTA_V1.0/train/labelTxt-v1.0/xml')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')

        temp_train_xml = os.listdir(train_xmlfilepath)
        train_xml = []
        for xml in temp_train_xml:
            if xml.endswith(".xml"):
                train_xml.append(xml)

        temp_val_xml = os.listdir(val_xmlfilepath)
        val_xml = []
        for xml in temp_val_xml:
            if xml.endswith(".xml"):
                val_xml.append(xml)

        temp_test_name = os.listdir(test_imgnamepath)
        test_imgname = []
        for img in temp_test_name:
            if img.endswith(".png"):
                test_imgname.append(img)

        num_train = len(train_xml)
        list_train = range(num_train)
        num_val = len(val_xml)
        list_val = range(num_val)
        num_test = len(test_imgname)
        list_test = range(num_test)
        num_trainval = num_train+num_val

        # tv = int(num * trainval_percent)
        # tr = int(tv * train_percent)
        # trainval = random.sample(list, tv)
        # train = random.sample(trainval, tr)

        # print("train and val size", tv)
        # print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in list_train:
            name = train_xml[i][:-4] + '\n'
            ftrain.write(name)

        for i in list_val:
            name = val_xml[i][:-4] + '\n'
            fval.write(name)

        for i in list_test:
            name = test_imgname[i][:-4] + '\n'
            ftest.write(name)

        for i in list_train:
            name = train_xml[i][:-4] + '\n'
            ftrainval.write(name)
        for i in list_val:
            name = val_xml[i][:-4] + '\n'
            ftrainval.write(name)
            # if i in trainval:
            #     ftrainval.write(name)
            #     if i in train:
            #         ftrain.write(name)
            #     else:
            #         fval.write(name)
            # else:
            #     ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')

            for image_id in image_ids:
                # list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                list_file.write('%s/DOTA_V1.0/%s/images/%s.png' % (os.path.abspath(VOCdevkit_DOTA_path), image_set, image_id))

                convert_annotation(year, image_id, list_file,image_set)
                list_file.write('\n')
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
