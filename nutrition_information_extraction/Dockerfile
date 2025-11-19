FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*


# Install pytest and any other dependencies
RUN pip install --no-cache-dir pytest==9.0.1
RUN pip install --no-cache-dir pytest-depends==1.0.1
RUN pip install --no-cache-dir litellm==1.80.0
RUN pip install --no-cache-dir pillow==12.0.0

WORKDIR /tests

# Copy tests and optionally source code if needed for imports
COPY test_*.py .

COPY images/* images/

# Set default command to run pytest
CMD ["pytest"]