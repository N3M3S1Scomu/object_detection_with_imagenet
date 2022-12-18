


# VGG16 modülü
from keras.applications.vgg16 import VGG16

"""
    we can make that with other CNN architectures
"""
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50,ResNet101,ResNet152

# görüntü önişleme
from keras.utils import img_to_array,load_img

# tahminleri basitçe yazdırabilmek için fonksiyonlar
from keras.applications.vgg16 import preprocess_input,decode_predictions

# giriş verilerimiz üzerine matris işlemleri yapmak için numpy
import numpy as np

# daha önce egitilmiş olan modelin agırlıklarını alıyoruz
model=VGG16(weights="imagenet") # hazır agırlıkları aldık

# goruntumuzun yolu
img_path="husky.jpg"

# goruntunun boyutlarını VGG16'ya uygun hale getiriyoruz
img=load_img(img_path,target_size=(224,224))

# boyutları ayarlanan goruntuyu matris dizisine ceviriyoruz
x=img_to_array(img)

# matrise cevirdigimiz goruntunun eksenlerini alıyoruz
x=np.expand_dims(x,axis=0)

# tahmine hazır hale getiriyoruz
x=preprocess_input(x)

# giriş orneklerimiz için çıktı tahminlerimizi üretiyoruz
preds = model.predict(x)

# goruntumuze en yakın 3 tahmini yazdırıyoruz
print("predicted:",decode_predictions(preds,top=3))




