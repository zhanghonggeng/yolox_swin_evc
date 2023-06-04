import cv2
import xml.etree.ElementTree as ET
import os

img_path = './dataset/DOTA_V1.0/train/images'
xml_path = './dataset/DOTA_V1.0/train/labelTxt-v1.0/xml'
save_path = './dataset/DOTA_V1.0/train/results'
colors = {'truck': (0, 0, 255), 'car': (0, 255, 0), 'people': (255, 0, 0)}

img_names = os.listdir(img_path)#以列表的形式获取文件夹中的所有文件的名字和格式（例如：0.jpg）
for img_name in img_names:
    img = os.path.join(img_path, img_name)#将文件的绝对路径与每个文件名字进行拼接，以获取该文件
    img = cv2.imread(img)#读取该文件（图片）
    xml_name = img_name.split('.')[0]#split（）分割文件路径，取分割后的第一个元素
    xml = os.path.join(xml_path, xml_name + '.xml')#拼接也可以直接使用+号
    xml.replace('\\','/')

    #读取xml文件
    xml_file = ET.parse(xml)
    root = xml_file.getroot()

    objects = root.findall('object')#查找所有名字为‘objects’的标签内容
    for obj in objects:
        obj_name = obj.find('name').text.strip()#查找名字为‘name'的标签内容
        xmin = int(float(obj.find('bndbox').find('xmin').text.strip()))#查找名字为‘bndbox’标签下的‘xmin’标签内容
        xmax = int(float(obj.find('bndbox').find('xmax').text.strip()))
        ymin = int(float(obj.find('bndbox').find('ymin').text.strip()))
        ymax = int(float(obj.find('bndbox').find('ymax').text.strip()))

        cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 0, 255))#画矩形，参数2和3是矩形的左上角点和右下角点的坐标
        cv2.putText(img, obj_name, (xmin, ymin-5), fontFace=cv2.CALIB_SAME_FOCAL_LENGTH,
                    fontScale=0.5, color=(0, 0, 255))#在图片上附上文字，字体和字号和颜色

    cv2.imshow('result', img)#显示
    cv2.waitKey(1000)#等待1000微秒
    cv2.imwrite(os.path.join(save_path, img_name), img)#将img写入到
