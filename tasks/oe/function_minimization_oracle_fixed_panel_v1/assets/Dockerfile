ARG PYTHON_IMAGE=python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
FROM ${PYTHON_IMAGE}
# Future scientific-library runtime profile1, Linux x86_64 CPython3.11 wheels from PyPI.
# The requirement and artifact hash are inside this assets-hashed Dockerfile.
LABEL sciel.oracle.dependency.profile="numpy-2.3.5-scipy-1.16.3-sklearn-1.7.2/1"
RUN printf '%s\n' 'numpy==2.3.5 --hash=sha256:8cba086a43d54ca804ce711b2a940b16e452807acebe7852ff327f1ecd49b0d4' 'scipy==1.16.3 --hash=sha256:0151a0749efeaaab78711c78422d413c583b8cdd2011a3c1d6c794938ee9fdb2' 'scikit-learn==1.7.2 --hash=sha256:89877e19a80c7b11a2891a27c21c4894fb18e2c2e077815bcade10d34287b20d' 'joblib==1.5.2 --hash=sha256:4e1f0bdbb987e6d843c70cf43714cb276623def372df3c22fe5266b2670bc241' 'threadpoolctl==3.6.0 --hash=sha256:43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb' > /tmp/oracle-numerics.txt \
    && python -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r /tmp/oracle-numerics.txt \
    && python -m pip check \
    && python -c 'import numpy, scipy, sklearn, joblib, threadpoolctl; from scipy.stats import norm; from sklearn.gaussian_process import GaussianProcessRegressor; assert (numpy.__version__, scipy.__version__, sklearn.__version__, joblib.__version__, threadpoolctl.__version__) == ("2.3.5", "1.16.3", "1.7.2", "1.5.2", "3.6.0"); assert norm.cdf(0.0) == 0.5' \
    && rm /tmp/oracle-numerics.txt
WORKDIR /opt/oracle
COPY container_supervisor.py oracle_client.py candidate_runner.py ./
RUN chmod 0555 /opt/oracle && chmod 0444 /opt/oracle/*.py && mkdir /input
ENTRYPOINT ["python", "-I", "-B", "/opt/oracle/container_supervisor.py"]
