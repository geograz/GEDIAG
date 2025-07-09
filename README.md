# GEDIAG

Repository for the code and analyses of the study on "Generational Dialogue in Geotechnics"

## Code description

An international online survey was conducted from March to July 2025 to explore generational differences in experiences, beliefs, needs, and goals of members of the geo-community. The survey targeted professionals independent of age and experience who study or work in geotechnics or other geo-related disciplines. The results aim to serve as a foundation for increasing the profession’s attractiveness, offer valuable insights into intergenerational biases, and lay the groundwork for improved collaboration in the workplace.

Survey Organizers: Alexander Kluckner, Georg Erharter, Andreas-Nizar Granitzer, Bettina Mair, Suzanne Lacasse


## Repository structure

- **figures/**: Contains all figures generated from the survey analyses.
    - Figures prefixed with `ALL_` display results for all survey responses.
    - Figures prefixed with `DACH_` display results specifically for responses from Austria :austria:, Germany :de:, and Switzerland :switzerland:.
- **src/**: Contains all source code used for data analysis and figure generation.
- **data/**: (Not included in the repository) Place the required dataset here to run analyses.
- **environment.yaml**: Lists all dependencies needed to reproduce the analysis environment.

This structure ensures clear separation between code, results, and data, making it easy to navigate and reproduce the study.


## Requirements

The environment is set up using `conda`.

To create the environment called `gediag` from the `environment.yaml` file, run:

```bash

conda env create --file environment.yaml
conda activate gediag

```

You may need to install the [Anaconda or Miniconda](https://www.anaconda.com/download/success) distribution beforehand to use conda commands.



