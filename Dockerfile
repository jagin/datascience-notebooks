FROM jupyter/datascience-notebook:1fbaef522f17

LABEL Jarosław Gilewski <jgilewski@jagin.pl>

USER root

RUN conda install -y pandas==0.20.3 && \
    pip install squarify==0.3.0

COPY .jupyter/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py
RUN chown -R $NB_USER $HOME/.jupyter
RUN chmod -R 755 $HOME/.jupyter