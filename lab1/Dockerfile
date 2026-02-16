FROM python:3.11-slim


WORKDIR /app

# Copy project files
COPY . /app

# Install Python deps
RUN pip install -r requirements.txt


# Run pipeline on container start
CMD ["/bin/bash", "./pipeline.sh"]
