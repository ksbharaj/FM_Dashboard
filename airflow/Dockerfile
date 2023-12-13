FROM quay.io/astronomer/astro-runtime:5.0.1

USER root

# Install system dependencies for Chrome
RUN apt-get update && apt-get install -y wget gnupg2
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update && apt-get install -y google-chrome-stable

# Additional flags sometimes necessary for headless Chrome
RUN apt-get install -y xvfb

# Copy only the requirements.txt at first to leverage Docker cache
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of your application code
COPY . /app
