FROM python
RUN pip install --upgrade pip
RUN pip install -r Karthikputchala/find-me/requirements.txt
CMD ['python','Karthikputchala/find-me/run.py']