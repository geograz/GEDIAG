# GEDIAG

Repository for the code and analyses of the study on "Generational Dialogue in Geotechnics"

## Code description

An international online survey was conducted from March to July 2025 to explore generational differences in experiences, beliefs, needs, and goals of members of the geo-community. The survey targeted professionals independent of age and experience who study or work in geotechnics or other geo-related disciplines. The results aim to serve as a foundation for increasing the profession’s attractiveness, offer valuable insights into intergenerational biases, and lay the groundwork for improved collaboration in the workplace.

Survey Organizers: Alexander Kluckner, Georg Erharter, Andreas-Nizar Granitzer, Bettina Mair, Suzanne Lacasse


## Repository structure

- All figures with analyses of questions can be found in the folder `figures`. Figures starting with "ALL_" show results for all submitted answers. Figures starting with "DACH_" show results for answers submitted from Austria :austria:, Germany :de: and Switzerland :switzerland:.
- The code for the analyses can be found in the folder `src`


## Requirements

The environment is set up using `conda`.

To create the environment called `gediag` from the `environment.yaml` file, run:

```bash

conda env create --file environment.yaml
conda activate gediag

```

Note: You may need to install the [Anaconda or Miniconda](https://www.anaconda.com/download/success) distribution beforehand to use conda commands.



