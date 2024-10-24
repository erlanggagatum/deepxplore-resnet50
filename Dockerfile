FROM python:3.5

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "gen_diff_resnet50-cifar10.py", "blackout", "0.5", "0.3", "7", "250", "250", "0.2"]