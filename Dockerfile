FROM python
RUN pip install --upgrade pip
RUN pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD ['python','Karthikputchala/find-me/run.py']