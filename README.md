# Code for variance-reduced Moment Generating Function with Control Variates
This repository contains the Python code used for the simulations in the paper:

> **Introducing LSECV: a variance-reduced LogSumExp estimator with Control Variates**  
> Louison Bocquet--Nouaille, Benjamin Bobbia, Philippe Naveau, Jérôme Morio




## Setup Virtual Environment (Optional)
It is recommended to use a virtual environment to isolate project dependencies.

> **Note:** These commands are suited for Windows.

To create a virtual environment named "env":
```bash
python -m venv env
```

To activate the virtual environment:
```bash
.\env\Scripts\activate
```

To deactivate the virtual environment:
```bash
deactivate
```



## Requirements

> **Note:** This project was developed and tested using **Python 3.12.7**.

Install dependencies with:
```bash
pip install -r requirements.txt
```



## Run code

Execute these commands from the main project folder:

* **Run simulation experiments:**

  Run with the default parameters:

  ```bash
  python -m src.experiment
  ```

  Run with custom parameters (example):

  ```bash
  python -m src.experiment --t_min 5 --t_max 100 --t_step 5 --n 500_000 --param_copula 5 --param_T 0.2 --param_S 0.3 --EVI -1.5 --EVI_diff 0.5 --endpoint 1 --endpoint_diff 0.4 --n_exp 10
  ```

The results will be saved in the `results` directory as `.csv` (for detailed results), `.png` (for quick preview) and `.eps` (as vector format).



## Repository structure

The *src/functions/create_samples.py* file provides the function to create samples with Generalized Pareto marginals and Gumbel copula.

The *src/functions/estimators.py* file provides the functions to compute the variance-reduced MGF estimators presented in the article (with exact and approximate control variates).

The *src/functions/plot.py* file provides the function to create the plots.

The *src/functions/save_to_csv* file provides the function to save the detailed results to a csv file.

The *src/experiment.py* file provides the code to run the expriment, saving the results to a csv and creating associated plots.


## License

This code is released under the MIT license (see file LICENSE).
