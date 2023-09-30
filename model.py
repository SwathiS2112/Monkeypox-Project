from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from os import listdir
from keras.utils import img_to_array
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

# set random seed
np.random.seed(42)

# root folder
root_dir="C:\Fold1\Train"


def convert_img_to_tensor(fpath):
    # read image
    img = cv2.imread(fpath)
    img = cv2.resize(img, (256, 256))

    # converts image to array
    res = img_to_array(img)

    return res


def get_img_data_and_label(root_dir):
    dire = listdir(root_dir)

    image_dataset = []
    image_label = []
    classes = []

    binary_label = []
    i = 0

    for subdir in dire:
        binary_label.append(i)
        classes.append(subdir)
        i += 1

    index = 0

    for subdir in dire:
        skin_img_list = listdir(f"{root_dir}/{subdir}")

        for imgfile in skin_img_list:
            filepath = f"{root_dir}/{subdir}/{imgfile}"
            # convert image to array
            res = convert_img_to_tensor(filepath)
            # add data to dataset list
            image_dataset.append(res)
            image_label.append(binary_label[index])

        index += 1

    return image_dataset, image_label, len(binary_label), classes


image_dataset,image_labels,NoOfOutputLayer,classes = get_img_data_and_label(root_dir)
print(NoOfOutputLayer)
print(len(image_labels))
print(len(image_dataset))
print(image_dataset[0].shape)
print(classes)

xtrain,xtest,ytrain,ytest=train_test_split(image_dataset,image_labels,test_size=0.2,random_state=100)

from keras.utils import to_categorical
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)



xtrain = np.array(xtrain, dtype=np.float16)/ 255.0
xtrain = xtrain.reshape(-1,256,256,3)
xtest = np.array(xtrain, dtype=np.float16)/ 255.0
xtest = xtrain.reshape(-1,256,256,3)


print(xtrain.shape)
print(ytrain.shape)

from keras.models import Sequential

model = Sequential()

model.add(Conv2D(32,(3,3), activation = 'relu' , input_shape = (256,256,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=32)
model.summary()
model.save('monkey_model.h5')