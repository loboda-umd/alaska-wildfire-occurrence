FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y git gcc build-essential python3-dev libgeos-dev

RUN python3 -m pip install --upgrade pip setuptools wheel

#RUN mamba install -c conda-forge leafmap geopandas localtileserver -y && \
#    fix-permissions "${CONDA_DIR}" && \
#    fix-permissions "/home/${NB_USER}"

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir --compile -r requirements.txt

ENV PROJ_LIB='/opt/conda/share/proj'

#USER root
#RUN chown -R ${NB_UID} ${HOME}
#USER ${NB_USER}

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME

RUN mkdir ./pages
COPY --chown=user /pages ./pages

EXPOSE 7860

CMD ["streamlit", "run", "./pages/Home.py", "--server.port=7860"]
