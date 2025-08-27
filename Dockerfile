FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip

RUN python3 -m pip install --upgrade pip && \
    PIP_ONLY_BINARY=:all: pip3 install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY app/ ./app/main_prod.py
# COPY fonts/ ./fonts/

CMD [ "app.main_prod.handler"]