FROM python:3.10

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    gfortran \
    python3-pkgconfig \
    libopenblas-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* --verbose

# Download and install Miniconda for ppc64le
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh -O ~/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash ~/miniconda.sh -b -f -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -i -l -t -p -y && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc
# Update PATH to include conda
ENV PATH $CONDA_DIR/bin:$PATH

# Copy the environment.yml file and create the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a
ENV PATH /opt/conda/envs/triagerxenv/bin:$PATH

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Copy the application files
COPY . .

EXPOSE 80

# Run the main.py script when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "triagerxenv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]