import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import joblib

# Loading the model
model = joblib.load('text-emotion-classifier.joblib')

# Predicting the emotions
def predict_emotions(docx):
    result = model.predict([docx])
    return result[0]

# Getting the prediction probabilities
def get_prediction_proba(docx):
    result = model.predict_proba([docx])
    return result

# Emotion emoji Dictionary
emotions_emoji_dict = {'surprise': '😲', 'love': '❤️',
                       'fear': '😨', 'anger': '😡',
                       'sadness': '😢', 'joy': '😄'
                       }

# Main Application
def main():
    st.title("Emotion Classifier App")
    
    with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type the statement here")
            submit_text = st.form_submit_button(label='Predict')
            
            if submit_text:
                 col1, col2 = st.columns(2)


                 # Apply Fxn Here
                 prediction = predict_emotions(raw_text)
                 probability = get_prediction_proba(raw_text)


                 with col1:
                      st.success("Original Text")
                      st.write(raw_text)
                      st.success("Prediction")
                      emoji_icon = emotions_emoji_dict[prediction]
                      st.write("{}:{}".format(prediction, emoji_icon))
                      st.write("Confidence:{}".format(np.max(probability)))


                 with col2:
                      st.success("Prediction Probability")
                      proba_df = pd.DataFrame(probability, columns=model.classes_)
                      proba_df_clean = proba_df.T.reset_index()
                      proba_df_clean.columns = ["emotions", "probability"]

                      fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                      st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
