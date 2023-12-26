FROM quay.io/astronomer/astro-runtime:5.0.1

USER root

# Install system dependencies for Chrome
RUN apt-get update && apt-get install -y wget gnupg2 unzip git
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update -y

# Additional flags sometimes necessary for headless Chrome
RUN apt-get install -y xvfb

# Set up Chromedriver Environment variables and install chrome
ENV CHROMEDRIVER_VERSION 114.0.5735.90
ENV CHROME_VERSION 114.0.5735.90-1
RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb \
  && apt install -y /tmp/chrome.deb \
  && rm /tmp/chrome.deb

ENV CHROMEDRIVER_DIR /chromedriver
RUN mkdir $CHROMEDRIVER_DIR

# Download and install Chromedriver
RUN wget -q --continue -P $CHROMEDRIVER_DIR "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
RUN unzip $CHROMEDRIVER_DIR/chromedriver* -d $CHROMEDRIVER_DIR

# Put Chromedriver into the PATH
ENV PATH $CHROMEDRIVER_DIR:$PATH

# Copy only the requirements.txt at first to leverage Docker cache
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install git+https://github.com/ksbharaj/ScraperFC.git@main

# Set the environment variable for GCP credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/airflow/gcp-key.json

ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Copy the GCP service account key file into the Docker container
COPY steel-cursor-408518-814d35f4b647.json /airflow/gcp-key.json

# Set Airflow logging level
ENV AIRFLOW__CORE__LOGGING_LEVEL=INFO

# Copy the rest of your application code
COPY . /app
