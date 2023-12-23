FROM python:3.9-slim
COPY . /app
COPY models/text-emotion-classifier.joblib /app/text-emotion-classifier.joblib
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit","run","app.py"]