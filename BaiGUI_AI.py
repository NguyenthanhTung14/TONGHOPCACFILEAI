import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import matplotlib.pyplot as plt
import cv2

def predict(url):
# Load mô hình
    model = load_model('C://TRITUENHANTAO//Baicuoiki//animals.h5')
    url = url.replace("/","//")
    # Load ảnh cần dự đoán và thực hiện tiền xử lý
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//cat.jpg', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//cow.jpg', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//dog.png', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//duck.jpg', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//elephant.jpg', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//horse.jpg', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//monkey.png', target_size=(40, 40))
    img = load_img(url, target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//sheep.png', target_size=(40, 40))
    #img = load_img('C://TRITUENHANTAO//Baicuoiki//Test_animals-20230516T042741Z-001//Test_animals//snake.jpg', target_size=(40, 40))

    img_array = img_to_array(img)
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán tên của ảnh
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    result = ['','cat', 'cow', 'dog', 'duck', 'elephant', 'horse', 'monkey', 'rabbit', 'sheep', 'snake']
    return result[class_index]
    # class_name = 'class_labels.txt'   # Thay đổi tên file dựa trên tên classes
    # # with open(class_name) as f:
    # #     class_labels = f.readlines()
    # #     predicted_class = class_labels[class_index]

    # Hiển thị tên và hình ảnh dự đoán được
    # plt.imshow(img)
    # plt.title(result[class_index])
    # plt.show()
