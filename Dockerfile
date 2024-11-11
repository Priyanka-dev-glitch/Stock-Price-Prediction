FROM python:3.7
WORKDIR /stockprediction
COPY src/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt 
COPY . .
EXPOSE 8181
ENTRYPOINT ["uvicorn", "src.app:app", "--reload", "--host", "0.0.0.0", "--port", "8181"]

