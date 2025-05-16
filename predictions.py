import numpy as np
import pandas as pd
import pm4py
from sklearn.preprocessing import LabelEncoder

def single_input_prediction(model,df,label_encoder,actual_label,option="train"):
    prediction = model.predict(df)
    indexes = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(indexes)
    print(f"{option} model recall {(actual_label[actual_label == predicted_label].shape[0]/actual_label.shape[0])*100}%")
    return predicted_label

def multi_input_prediction(model,df_1,df_2,label_encoder,actual_label,option="train"):
    prediction = model.predict([df_1,df_2])
    indexes = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(indexes)
    print(f"{option} model recall {(actual_label[actual_label == predicted_label].shape[0]/actual_label.shape[0])*100}%")
    return predicted_label