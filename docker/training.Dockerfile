FROM python:3.10

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]