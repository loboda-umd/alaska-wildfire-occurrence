FROM jupyter/base-notebook:latest

RUN mamba install -c conda-forge leafmap geopandas localtileserver -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir ./pages
COPY /pages ./pages

ENV PROJ_LIB='/opt/conda/share/proj'

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

EXPOSE 7860

CMD ["streamlit", "run", "./pages/Home.py", "--server.port=7860"]


streamlit run pages/Home.py