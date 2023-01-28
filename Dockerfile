FROM python
RUN pip install --upgrade pip
RUN pip freeze > requirements.txt
RUN find . -regex '.*requirements.txt$'
RUN pip install -r requirements.txt
CMD ['python','Karthikputchala/find-me/run.py']