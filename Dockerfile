# 1. Base Image: NVIDIA PyTorch container with CUDA and Python
FROM nvcr.io/nvidia/pytorch:22.12-py3

# 2. Set working directory
WORKDIR /app

# 3. Upgrade pip and install gdown
RUN pip install --upgrade pip && pip install gdown

# 4. Copy the requirements file
COPY ./LineVul/requirements.txt /app/requirements.txt

# 5. Create a requirements file for the Docker environment.
# Exclude/downgrade packages for Python 3.8 compatibility.
RUN grep -v 'torch' requirements.txt | grep -v 'nvidia' | \
    sed 's/matplotlib==3.9.0/matplotlib==3.5.3/' | \
    sed 's/contourpy==1.2.1/contourpy==1.0.7/' | \
    sed 's/scipy==1.14.0/scipy==1.10.1/' > requirements.docker.txt

# 6. Install the Python dependencies
RUN pip install -r requirements.docker.txt

# 7. Copy the rest of the application code
COPY ./LineVul /app/

# 8. Download the big-vul dataset into the image
RUN gdown "https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V" -O /app/data/big-vul_dataset/test.csv && \
    gdown "https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw" -O /app/data/big-vul_dataset/train.csv && \
    gdown "https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ" -O /app/data/big-vul_dataset/val.csv

# 9. Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]

