import gradio as gr
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

#Dictionary to replace 
res_dict = {
    0: "Mild dementia detected",
    1: "Moderate dementia detected",
    2: "No dementia detected",
    3: "Very mild dementia detected"
}

def classify_image(img):
    #Preprocessing image
    scan = tf.keras.preprocessing.image.img_to_array(img) 
    scan = scan.reshape(-1,128, 128,3)
    LeNet5 = load_model('LeNet-5e50Mod.h5')
    return res_dict[np.argmax(LeNet5.predict(scan))]


app = gr.Interface(title="Input your horizontal section MRI scan",
            fn=classify_image, 
             inputs=gr.inputs.Image(shape=(128, 128)),
             outputs=gr.outputs.Label(num_top_classes=4),
             examples=["output/test/Mild_Demented/mild_2.jpg", #Mild dementia CT Scan
                        "output/test/Moderate_Demented/moderate_7.jpg", #Moderate dementia CT Scan
                        "output/test/Non_Demented/non_10.jpg", #No dementia CT Scan
                        "output/test/Very_Mild_Demented/verymild_3.jpg" #Very mild dementia CT Scan
                        ] 
                        )
             
app.launch()