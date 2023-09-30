import tensorflow as tf
from flask import Flask,render_template, request
from keras.utils import load_img
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

app = Flask(__name__)



@app.route('/',methods=['POST','GET'])
def index():

    model = load_model('monkey_model.h5')

    # for check the image |Testng

    def check(res):
            path = ['Monkey pox','Others']
            pred = model.predict(res)
            res = np.argmax(pred)
            res = path[res]
            return (res)


    def convert_img_to_tensor2(fpath):
        img = cv2.imread(fpath)
        img = cv2.resize(img, (256, 256))
        res = img_to_array(img)
        res = np.array(res, dtype=np.float16) / 255.0
        res = res.reshape(-1, 256, 256, 3)
        res = res.reshape(1, 256, 256, 3)
        return res



    if request.method == 'POST':
        img = request.files['img']
        img.save('static/example.jpg')
        res = convert_img_to_tensor2("static/example.jpg")
        msg = check(res)
        return render_template('result.html', res=msg)

    else:
        return render_template('mfront.html', res="invalid input")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")