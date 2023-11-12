from skimage.filters import gaussian  #pip install scikit-image
from keras.models import *
import numpy as np


def myGaussSmooth( data, std=1.5):
    return gaussian(data,sigma=std,truncate=2)
def clip(data, vmin, vmax):
  return np.clip(data,vmin,vmax)

def normalize(data):
  return 2*((data - np.min(data))/np.max(data))-1

# Loading model and weight
# load json and create model
json_file = open('./ModelandWeight/Fault_Detection_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./ModelandWeight/Fault_Detection_Weight.h5")
print("Loaded model from disk")


def Predictor3(img, model, patch_size):
    x, y = img.shape
    img = myGaussSmooth(img)
    v = np.percentile(img, 99)
    img = normalize(clip(img, -v, v))
    if x % 16 == 0 and y % 16 == 0:
        horizontal_patch = y / patch_size
        vertical_patch = x / patch_size
        patch = np.full_like(img, 0)
        probability = np.zeros((x, y, 2))

        for xcor in range(int(horizontal_patch)):
            for ycor in range(int(vertical_patch)):
                img_picked = img[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                             int(patch_size * ycor):int(patch_size * (ycor + 1))]
                xp, yp = img_picked.shape
                patch[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                int(patch_size * ycor):int(patch_size * (ycor + 1))] = np.argmax(
                    model.predict(np.reshape(img_picked, (1, xp, yp, 1))), axis=-1)
                probability[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                int(patch_size * ycor):int(patch_size * (ycor + 1)), :] = np.squeeze(
                    model.predict(np.reshape(img_picked, (1, xp, yp, 1))))
        return patch, probability

    else:
        x_remain = x // 8
        y_remain = y // 8
        padding_img = np.zeros(((x_remain + 1) * 8, (y_remain + 1) * 8))
        padding_img[0:x, 0:y] = img
        x_pad, y_pad = padding_img.shape
        horizontal_patch = x_pad / patch_size
        vertical_patch = y_pad / patch_size
        patch = np.full_like(padding_img, 0)
        probability = np.zeros((x_pad, y_pad, 2))
        for xcor in range(int(horizontal_patch)):
            for ycor in range(int(vertical_patch)):
                img_picked = padding_img[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                             int(patch_size * ycor):int(patch_size * (ycor + 1))]
                xp, yp = img_picked.shape
                patch[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                int(patch_size * ycor):int(patch_size * (ycor + 1))] = np.argmax(loaded_model.predict(np.reshape(img_picked, (1, xp, yp, 1))), axis=-1)
                probability[int(patch_size * xcor):int(patch_size * (xcor + 1)),
                int(patch_size * ycor):int(patch_size * (ycor + 1)), :] = np.squeeze(model.predict(np.reshape(img_picked, (1, xp, yp, 1))))
        return patch[0:x, 0:y], probability[0:x, 0:y, :]

