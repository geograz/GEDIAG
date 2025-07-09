# -*- coding: utf-8 -*-
"""
Code to analyze the survey for
---- GENERATIONAL DIALOGUE IN GEOTECHNICS ----

Script executes the analyses and creates plots for publications

Code author: Georg Erharter
georg.erharter@ngi.no
"""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from dictionaries import dicts
from library import Plotter, Preprocessor, Utilities


#############################
# fixed variables and constants
#############################

FN = r'Generational dialogue in geotechnics(1-822)' # File name of the survey results
GENERATIONS = ['Silent', 'Baby Boomers', 'X', 'Y', 'Z'] # "Silent" not used in the survey, but included for completeness
MOD = 'ALL'  # modifier for filenames  # DACH, ALL


#############################
# loading and preprocessing of data
#############################

# get the directory where the current python file is
script_dir = os.path.dirname(os.path.abspath(__file__))

folders = [#r'C:\Users\GEr\Dropbox\Apps\Overleaf\GEDIAG_FactualReport\figures',
           os.path.abspath(os.path.join(script_dir, '..', 'figures'))
           ]

print(folders)

# load dictionary, utilities and plotter class
d = dicts()
pltr = Plotter(GENERATIONS[1:], folders)
utils = Utilities()
prep = Preprocessor()

# read data from excel file
file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', f'{FN}.xlsx'))
df = pd.read_excel(file_path, keep_default_na=False)

# rename columns where necessary
df.rename(d.questions, axis=1, inplace=True)

# replace answers that are unusable in raw form with values from dictionary
df['country'] = df['country'].replace(d.countries)
# TODO: might be reaosonable to transparently address the mapping of professions and primary work fields
df['profession'] = df['profession'].replace(d.professions)
df['primary field of work'] = df['primary field of work'].replace(d.primary_work_field)

# replace answers that are unusable in raw form with "other"
for q_nr in [8, 19, 20, 21, 22, 23, 25, 30]:
    df = prep.replace_all_but(df, column=d.question_numbers[q_nr],
                              exceptions=d.answers[q_nr])

# DEBUG: print unique values of resepctive columns after processing
for q_nr in [8, 19, 20, 21, 22, 23, 25, 30]:
    print(f"Unique values in {q_nr}: {df[d.question_numbers[q_nr]].unique()}")

# process multiple choice answers
for q_nr in [14, 24, 27, 32, 34]:
    # make new columns in dataframe for up to 3 answers
    for choice in [1, 2, 3]:
        df[f'{q_nr}_choice_{choice}'] = ''
    
    # iterate through rows to assign answer possibilities to columns
    for i, row in df.iterrows():
        try:
            row = row[d.question_numbers[q_nr]].split(';')[:-1]
            row = [a if a in d.answers[q_nr] else 'other' for a in row]
            # Limit to 3 choices (avoid > 3 colums in case other include semi-colon seperated entries)
            for j, c in enumerate(row[:3]):
                df.loc[i, f'{q_nr}_choice_{j+1}'] = c
        except AttributeError:
            pass

# replace '' with nan in questions
df = df.replace('', np.nan).infer_objects(copy=False)

# make sure that questions 31, 32, 33 and 34 logic is correct
for q_nr in [31, 33]:
    ids1 = list(df[df[d.question_numbers[q_nr]] == 'No (-> please skip next question)'].index)
    ids2 = list(df.loc[pd.isna(df[d.question_numbers[q_nr]]), :].index)
    ids = ids1 + ids2
    df.loc[ids, d.question_numbers[q_nr+1]] = np.nan

# switch to include only DACH countries if requested
if MOD == 'DACH':
    df = df[(df['country'] == 'Austria') |
            (df['country'] == 'Germany') |
            (df['country'] == 'Switzerland')]


#############################
# analyses
#############################

# compute birth year from survey completion time and age
df['birth year'] = (df['Completion time'] - pd.to_timedelta(
    df['age [y]'] * 365, unit='D')).dt.year

# make dataframe with analyses per generation
df_generations = pd.DataFrame(index=GENERATIONS)
df_generations['from'] = [1928, 1946, 1965, 1981, 1997]
df_generations['to'] = [1945, 1964, 1980, 1996, 2012]

# assign participants to generations
df['generation'] = '-'
for g in df_generations.index:
    id_ = df[(df['birth year'].values >= df_generations['from'].loc[g]) &
             (df['birth year'].values <= df_generations['to'].loc[g])].index
    df.loc[id_, 'generation'] = g

# get total submission numbers for genders
df_generations['n submissions'] = df.groupby('generation').size()
df_genders = df.groupby(['generation', 'gender']).size().unstack()
df_genders.columns = [f'n {g}' for g in df_genders.columns]

count_cols = [col for col in df_genders.columns if col.startswith('n ')] # true gender counts are used as the denominator
for gender in ['Male', 'Female', 'Non-binary']:
    col = f'n {gender}'
    if col in df_genders.columns:
        df_genders[f'% {gender}'] = df_genders[col] / df_genders[count_cols].sum(axis=1)

df_generations = pd.concat((df_generations, df_genders), axis=1)
df_generations.fillna(0, inplace=True)

#############################
# print selected stats
#############################

print('n participants', len(df))
print('n participants AT:', len(df[df['country'] == 'Austria']))
print('n participants DE:', len(df[df['country'] == 'Germany']))
print('n participants CH:', len(df[df['country'] == 'Switzerland']))
print('age range:', df['birth year'].min(), df['birth year'].max())
print('genders:', np.unique(df['gender'], return_counts=True), '\n')

utils.relative_numbers(df, 'profession')
utils.relative_numbers(df, 'primary field of work')
utils.relative_numbers(df, 'relevant work experience [y]')
utils.relative_numbers(df, 'organization size')
utils.relative_numbers(df, 'position')
utils.relative_numbers(df, 'Do you have responsibility for personnel?')
utils.relative_numbers(df[(df['generation'] == 'Baby Boomers') |
                          (df['generation'] == 'X')],
                       'Do you have responsibility for personnel?')
utils.relative_numbers(df[(df['generation'] == 'Y') |
                          (df['generation'] == 'Z')],
                       'Do you have responsibility for personnel?')

utils.relative_numbers(df, d.question_numbers[16])
utils.relative_numbers(df, d.question_numbers[17])
utils.relative_numbers(df, d.question_numbers[20])
utils.relative_numbers(df, d.question_numbers[22])
utils.relative_numbers(df[(df['generation'] == 'Baby Boomers') |
                          (df['generation'] == 'X')],
                       d.question_numbers[23])
utils.relative_numbers(df[(df['generation'] == 'Y') |
                          (df['generation'] == 'Z')],
                       d.question_numbers[23])
utils.relative_numbers(df, d.question_numbers[25])
utils.relative_numbers(df, d.question_numbers[30])
utils.relative_numbers(df, d.question_numbers[42])


#############################
# plotting
#############################

pdf_path = os.path.abspath(os.path.join(script_dir, '..', '..',
                                        f'{MOD}_combined_plots.pdf'))

with PdfPages(pdf_path) as pdf:
    # ---- Participant Data analyses

    pltr.submissions_age_histogram(df, df_generations,
                                   f'{MOD}_hist_n_submissions')
    fig = plt.gcf()
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # generations - gender stacked barchart
    pltr.submissions_barchart(df_generations,
                              f'{MOD}_bar_n_submissions')
    fig = plt.gcf()
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # professions bar chart plot
    pltr.bar_chart_generic(df, column='profession', x_sort='ascending',
                           filename=f'{MOD}_bar_professions')
    fig = plt.gcf()
    pdf.attach_note('question number 4')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # primary field of work bar chart plot
    pltr.bar_chart_generic(df, column='primary field of work',
                           x_sort='ascending',
                           filename=f'{MOD}_bar_primary_workfield')
    fig = plt.gcf()
    pdf.attach_note('question number 5')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # work experience bar chart plot
    pltr.bar_chart_generic(df, column='relevant work experience [y]',
                           filename=f'{MOD}_bar_work_experience',
                           x_sort=['< 5', '5-10', '11-20', '21-30', '> 30'])
    fig = plt.gcf()
    pdf.attach_note('question number 6')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # people in organization bar chart plot
    pltr.bar_chart_generic(df, column='organization size',
                           filename=f'{MOD}_bar_organization_size',
                           x_sort=['1 (sole proprietorship)', '2-9', '10-49',
                                   '50-249', '> 249', 'Not applicable'])
    fig = plt.gcf()
    pdf.attach_note('question number 7')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # position bar chart plot
    pltr.bar_chart_generic(df, column='position', x_sort='ascending',
                           filename=f'{MOD}_bar_position')
    fig = plt.gcf()
    pdf.attach_note('question number 8')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # participation world map
    pltr.participation_world_map(df, f'{MOD}_map_n_submissions')
    fig = plt.gcf()
    pdf.attach_note('question number 10')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # participation bar chart
    pltr.bar_chart_generic(df, column='country', x_sort='ascending',
                           max_n_bars=10,
                           filename=f'{MOD}_bar_n_submissions_country')
    fig = plt.gcf()
    pdf.attach_note('question number 10')
    pdf.savefig(fig)  # Saves the current figure into the PDF
    plt.close(fig)    # Close the figure to free memory

    # ---- Technical question analysis ---

    # rating questions
    print('rating questions')
    for q_nr in [11, 29, 38]:
        pltr.generation_rating_plot(df, d.rating_columns[q_nr],
                                    d.answers[q_nr], d.question_numbers[q_nr],
                                    f'{MOD}_{q_nr}_rating')
        fig = plt.gcf()
        pdf.attach_note(f'question number {q_nr}')
        pdf.savefig(fig)  # Saves the current figure into the PDF
        plt.close(fig)    # Close the figure to free memory

    # make violin plots for numerical questions
    print('violins')
    for q_nr in [12, 26, 28, 41, 45]:
        pltr.generation_violin_generic(df, d.question_numbers[q_nr],
                                       f'{MOD}_{q_nr}_violin')
        fig = plt.gcf()
        pdf.attach_note(f'question number {q_nr}')
        pdf.savefig(fig)  # Saves the current figure into the PDF
        plt.close(fig)    # Close the figure to free memory

    # make stacked barcharts for single choice questions
    print('single choice barcharts')
    for q_nr in [9, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 30, 35, 36, 37,
                 39, 40, 42, 43, 44, 46]:
        pltr.generation_perc_bar_generic(df, d.question_numbers[q_nr],
                                         d.answers[q_nr],
                                         filename=f'{MOD}_{q_nr}_perc_bar')
        fig = plt.gcf()
        pdf.attach_note(f'question number {q_nr}')
        pdf.savefig(fig)  # Saves the current figure into the PDF
        plt.close(fig)    # Close the figure to free memory

    # multiple choice questions
    print('multiple choice barcharts')
    for q_nr in [14, 24, 27, 32, 34]:
        # pltr.generation_perc_bar_multichoice(df, d.question_numbers[q_nr],
        #                                      q_nr, d.answers[q_nr],
        #                                      filename=f'{MOD}_{q_nr}_perc_bar_multi')
        pltr.generation_multichoice_hbars(df, d.question_numbers[q_nr],
                                          q_nr, d.answers[q_nr],
                                          filename=f'{MOD}_{q_nr}_perc_hbars_multi')
        fig = plt.gcf()
        pdf.attach_note(f'question number {q_nr}')
        pdf.savefig(fig)  # Saves the current figure into the PDF
        plt.close(fig)    # Close the figure to free memory

# %%
