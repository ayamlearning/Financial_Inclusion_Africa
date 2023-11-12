FROM python:3.10.12-slim

WORKDIR /app

COPY ["predict.py","requirements.txt","model_financial_inclusion.bin","./"]

RUN pip install -r requirements.txt

EXPOSE 8000

CMD python ./predict.py
