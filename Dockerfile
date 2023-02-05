FROM python:3.9
COPY . /usr/app/
EXPOSE 8080
WORKDIR /usr/app/
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
CMD python main.py
