FROM python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
RUN python -m pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 mpmath==1.3.0 sympy==1.14.0 networkx==3.4.2
RUN python -m pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir transformers==5.17.0
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf TOKENIZERS_PARALLELISM=false
LABEL org.sciel.runtime="repeat-circle/1"
WORKDIR /tmp
