FROM python:3.12.2

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . ./

# Expose port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "main.py", "--server.port", "8501"]