FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

COPY . .

# CMD ["python", "app.py"]
# CMD ["./app/finance_analysis.py"]

EXPOSE 8000

ENV PYTHONPATH=/app

# ENTRYPOINT ["python3"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

