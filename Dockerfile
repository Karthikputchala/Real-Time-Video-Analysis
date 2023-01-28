FROM python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ['python','Karthikputchala/find-me/run.py']