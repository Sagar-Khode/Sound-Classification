import os

import streamlit as st

import numpy as np
from pydub import AudioSegment


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

import librosa
import librosa.display
import random
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import soundfile as sf





model = load_model("model_CNN.hdf5")

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366")
    }
   .sidebar .sidebar-content {
        background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<h1 style='text-align: center; color: red;font-size:60px;margin-top:-50px;'>AUDIO CLASSIFICATION</h1><h1 style='text-align: center; color: black;font-size:30px;margin-top:-30px;'>Using Deep Learning</h1>""",
    unsafe_allow_html=True)


radio = st.sidebar.radio("Select format of audio file", options=['mp3', 'wav'])

if radio == 'wav':

        file = st.sidebar.file_uploader("Upload Audio To Classify", type=["wav"])

        if file is not None:
            st.markdown(
                """<h1 style='color:black;'>Audio : </h1>""",
                unsafe_allow_html=True)
            st.audio(file)

            rad = st.sidebar.radio("Choose Options", options=["Predict", "Spectrogram"])
            if rad == "Predict":
                if st.button("Predict Audio"):
                    
                    max_pad_len = 300
                    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    pad_width = max_pad_len - mfccs.shape[1]
                    mfccs_scaled_features = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    mfccs_scaled_features = mfccs_scaled_features.reshape((1 ,40,300,1))

                    predicted_label=model.predict(mfccs_scaled_features)

                    classes_x=np.argmax(predicted_label,axis=1)


                    if (classes_x == 0):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Air_conditioner <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==1):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Car_horn <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==2):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Children_playing <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==3):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Dog_bark <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==4):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Drilling <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==5):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Engine_idling <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==6):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Gun_shot <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==7):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Jackhammer <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==8):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Siren <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==9):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Street_music <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Spectrogram":


                samples, sample_rate = librosa.load(file)
                fig = plt.figure(figsize=[0.72,0.72])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))


                
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)


elif radio == 'mp3':
        file = st.sidebar.file_uploader("Upload Audio To Classify", type="mp3")

        if file is not None:
            sound = AudioSegment.from_mp3(file)
            sound.export("file.wav", format="wav")
            st.markdown(
                """<h1 style='color:yellow;'>Audio : </h1>""",
                unsafe_allow_html=True)
            a = st.audio(file, format="audio/mp3")

            rad = st.sidebar.radio("Choose Options", options=["Predict", "Spectrogram"])
            if rad == "Predict":
                if st.button("Classify Audio"):
                    
                    max_pad_len = 300
                    audio, sample_rate = librosa.load("file.wav", res_type='kaiser_fast') 
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    pad_width = max_pad_len - mfccs.shape[1]
                    mfccs_scaled_features = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    mfccs_scaled_features = mfccs_scaled_features.reshape((1 ,40,300,1))

                    predicted_label=model.predict(mfccs_scaled_features)

                    classes_x=np.argmax(predicted_label,axis=1)


                    if (classes_x == 0):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Air_conditioner <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==1):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Car_horn <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==2):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Children_playing <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==3):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Dog_bark <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==4):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Drilling <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==5):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Engine_idling <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==6):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Gun_shot <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==7):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Jackhammer <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==8):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Siren <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

                    elif (classes_x ==9):
                      st.markdown(
                        f"""<h1 style='color:skyblue;'>I am Predicted as : Street_music <span style='color:white;'></span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Spectrogram":


                samples, sample_rate = librosa.load(file)
                fig = plt.figure(figsize=[0.72,0.72])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))


                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:purple;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

            os.remove("file.wav")
