FROM agrigorev/zoomcamp-model:2025

# Install FastAPI, uvicorn, requests, and scikit-learn
RUN pip install fastapi uvicorn requests scikit-learn==1.6.1

# Copy the pipeline file and FastAPI script
COPY pipeline_v1.bin main.py ./

# Expose port 8000
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
