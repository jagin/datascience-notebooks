# datascience-notebooks
Collection of Data science/Machine Learning/Deep Learning Jupyter notebooks

| Notebook      | Language      | Description   |
| ------------- |:-------------:|:-------------:|
| [Console Games Sales Decline](http://nbviewer.jupyter.org/github/jagin/datascience-notebooks/blob/master/notebooks/Python/console-games-sales-decline.ipynb) | Python | Investigate the state of the game industry to help the client make the decision of whether to get into this business. |
| [Philadelphia Crime Rates](http://nbviewer.jupyter.org/github/jagin/datascience-notebooks/blob/master/notebooks/R/philadelphia-crime-rates.ipynb) | R | Find any helpful trends in Crime rates to assist the Philadelphia police department in planning their work for 2017. |

## Running Jupyter

With [Docker](https://www.docker.com/community-edition) you can quickly setup Jupyter environment to run the notebooks and do your own explorations.  
For more details see [opinionated stacks of ready-to-run Jupyter applications in Docker](https://github.com/jupyter/docker-stacks).

To run the datascience-notebook container run the following command:

```
docker-compose up --build
```

Wait for:

```
...
datascience-notebook    |     Copy/paste this URL into your browser when you connect for the first time,
datascience-notebook    |     to login with a token:
datascience-notebook    |         http://localhost:8888/?token=your_token
```

to be displayed on your console and follow the instruction.

If you need to run some additional commands in the container run:

```
bash -c clear && docker exec -it datascience-notebook sh

```