FROM python:3.9

RUN useradd -m -u 1000 user
USER user

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["streamlit", "run", "app.py"]
