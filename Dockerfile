# =====================================================================
# Loom AI - Document Retriever | Dockerfile
# =====================================================================
# Builds the Document Retriever service as a lightweight container.
#
# Build:  docker build -t loom-ai-doc-retriever .
# Run:    docker run -p 8001:8001 --env-file .env.live loom-ai-doc-retriever
# =====================================================================

FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (Docker layer caching optimization)
# This layer only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the service port (matches DOC_RETRIEVER_PORT)
EXPOSE 8001

# Start the FastAPI server using the app entry point
CMD ["python", "-m", "app.main"]
