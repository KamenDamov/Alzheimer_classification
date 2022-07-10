import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Reshape
from keras.models import load_model

def classify_image(input):
    input = input.Reshape((128, 128, 3))
    input = input.Rescaling(1./255)
    model = load_model('LeNet-5.h5')
    print(model.predict(input))
    return model.predict(input)


app = gr.Interface(fn=classify_image, 
             inputs=gr.inputs.Image(shape=(224, 224)),
             outputs=gr.outputs.Label(num_top_classes=3))
             
app.launch()