FROM python:3.8.3-slim

# Set up and activate virtual environment
ENV VIRTUAL_ENV "/venv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80

COPY model/model.tar.gz .
RUN tar xzf model.tar.gz  && rm model.tar.gz
COPY evaluate.py .
COPY main.py .
CMD ["python","./evaluate.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
