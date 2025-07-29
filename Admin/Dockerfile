FROM python:3.11
EXPOSE 8080
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit","run","admin.py","--server.port=8080" ] 