FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*


# Install pytest and any other dependencies
RUN pip install --no-cache-dir pytest==9.0.1
RUN pip install --no-cache-dir pytest-depends==1.0.1
RUN pip install --no-cache-dir pytest-repeated==0.3.4.dev202512071416
RUN pip install --no-cache-dir litellm==1.80.0
RUN pip install --no-cache-dir pillow==12.0.0
RUN pip install --no-cache-dir pillow-heif==1.1.1

WORKDIR /tests


CMD ["pytest", "--disable-warnings", "-vv"]