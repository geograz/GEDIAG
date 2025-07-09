# -*- coding: utf-8 -*-
"""
Code to analyze the survey for
---- GENERATIONAL DIALOGUE IN GEOTECHNICS ----

Script contains a custom library with functions that are required to analyze
and visualize the survey results.

Code author: Georg Erharter
georg.erharter@ngi.no
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# TODO: Might be more convenient to turn this into a @staticmethod as no state is shared
class Preprocessor:
    '''class with functions to preprocess the survey responses'''

    def __init__(self):
        pass

    def replace_all_but(self, df: pd.DataFrame, column: str, exceptions: list,
                        replace_with: str = 'other') -> pd.DataFrame:
        '''function replaces all answers in one column with "other"
        except for the predefined options documented in the exceptions list'''
        mask = ~df[column].isin(exceptions) & df[column].notna()
        df.loc[mask, column] = replace_with
        return df

# TODO: Might be more convenient to turn this into a @staticmethod as no state is shared
class Utilities:
    '''class with miscellaneous functions'''

    def __init__(self):
        pass

    def relative_numbers(self, df, column):
        '''function prints percentage of distinct answers for one specific question'''
        vals, counts = np.unique(df[column].astype(str), return_counts=True)
        counts = list(np.round(counts/counts.sum()*100, 0))
        dic = {'values': vals, 'perc': counts}
        dic = pd.DataFrame.from_dict(dic)
        print(column, dic, '\n')


class Plotter:
    '''class with plotting functions for survey analyses'''

    def __init__(self, GENERATIONS: list, FOLDERS: list):
        self.GENERATIONS = GENERATIONS
        self.FOLDERS = FOLDERS
        self.grey_colors = ['0.8', '0.6', '0.4', '0.2', '0.0'] * 4
        self.c_colors = ['indianred', 'slategrey', 'gainsboro',
                         'firebrick', 'steelblue', 'dimgrey',
                         'lightcoral', 'lightskyblue', 'grey',
                         'rosybrown', 'lightsteelblue', 'silver',
                         ] * 4

    def wrap_text(self, text: str, max_length: int = 60) -> str:
        '''function that wraps text to a maximum number of characters'''
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            # If adding the next word exceeds the max length, start a new line
            if len(current_line) + len(word) + (1 if current_line else 0) > max_length:
                lines.append(current_line)
                current_line = word
            else:
                current_line += (" " if current_line else "") + word

        if current_line:
            lines.append(current_line)

        return '\n'.join(lines)

    def plot_end(self, filename: str) -> None:
        '''internal conveniance function that is put at the end of most plots'''
        plt.tight_layout()
        for folder in self.FOLDERS:
            plt.savefig(fr'{folder}\{filename}.pdf')
            plt.savefig(fr'{folder}\{filename}.jpg', dpi=600)

    def submissions_age_histogram(self, df: pd.DataFrame,
                                  df_generations: pd.DataFrame, filename: str):
        '''histogram that shows numbers of submissions over birth years'''
        fig, ax = plt.subplots(figsize=(5, 4))

        bins = np.arange(df['birth year'].min(), df['birth year'].max() + 2) # Explicitly define the bin edges, including one extra year
        h = ax.hist(df['birth year'], bins=bins, color=self.c_colors[0], 
                    edgecolor='black')

        print(df['birth year'].max())
        ax.vlines(x=df_generations['from'], ymin=0, ymax=max(h[0])+2,
                  color='black', lw=2)
        for g in df_generations.index:
            if g == 'Baby Boomers':
                text = 'Baby\nBoomers'
            else:
                text = g
            ax.text(x=df_generations.loc[g]['from']+1, y=max(h[0])+1.5, s=text,
                    va='top')

        ax.set_ylim(top=max(h[0])+2)

        ax.grid(alpha=0.5, axis='y')
        ax.set_xlabel('birth year')
        ax.set_ylabel("number of participants")

        self.plot_end(filename)

    def submissions_barchart(self, df_generations: pd.DataFrame,
                             filename: str):
        '''barchart that shows how many submissions per generation were received'''
        fig, ax = plt.subplots(figsize=(5, 4))

        bottom = np.zeros(len(df_generations))
        x = np.arange(len(df_generations))

        for i, gender in enumerate(['n Female', 'n Male', 'n Non-binary']):
            gender_name = gender.split(' ')[1]
            try:
                ax.bar(x, df_generations[gender],
                       width=0.7,
                       label=gender_name,
                       bottom=bottom, color=self.c_colors[i])
                bottom += df_generations[gender].values

                for j, x_ in enumerate(x):
                    perc = int(round(df_generations[f"% {gender_name}"].iloc[j]*100))
                    if perc > 1:
                        ax.text(x=x_, y=bottom[j]+1, s=f'{perc}%', ha='center')
            except KeyError:
                pass

        ax.grid(alpha=0.5, axis='y')

        ax.set_xlabel("generation")
        ax.set_ylabel("number of participants")
        ax.set_xticks(x)
        ax.set_xticklabels(df_generations.index, rotation=0)
        ax.legend()

        self.plot_end(filename)

    def bar_chart_generic(self, df: pd.DataFrame, column: str,
                          filename: str, x_sort: str = 'descending',
                          max_n_bars: int = None) -> None:
        '''generic bar chart plot that should work for several questions'''
        fig, ax = plt.subplots(figsize=(5, 5))

        labels, counts = np.unique(df[column].dropna().astype(str),
                                   return_counts=True)

        if x_sort == 'descending':
            sorting = np.flip(np.argsort(counts))
        elif x_sort == 'ascending':
            sorting = np.argsort(counts)
        else:  # custom sorting allows to filter e.g. "other" answers
            arr_index_map = {value: idx for idx, value in enumerate(labels)}
            sorting = np.array([arr_index_map[value] for value in x_sort])

        # sort labels and evtl. reduce number of bars
        if max_n_bars is not None and x_sort == 'ascending':
            labels = labels[sorting][-max_n_bars:]
            counts = counts[sorting][-max_n_bars:]
        else:
            labels, counts = labels[sorting], counts[sorting]

        ax.barh(np.arange(len(labels)), counts, label=labels,
                color=self.c_colors[0])
        ax.grid(alpha=0.5, axis='x')

        labels = [self.wrap_text(l, max_length=30) for l in labels]
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, va='center')
        ax.set_ylabel(column)
        ax.set_xlabel('number of submissions')

        self.plot_end(filename)

    def participation_world_map(self, df: pd.DataFrame, filename: str) -> None:
        '''plots a world map that shows the participation numbers globally'''
        # Load world map
        # url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        # world = gpd.read_file(url)

        world = gpd.read_file(
            r'world-administrative-boundaries\world-administrative-boundaries.shp')

        world.to_excel('world.xlsx')

        labels, counts = np.unique(df['country'], return_counts=True)

        data = pd.DataFrame({'country': labels, 'value': counts})
        data['categories'] = np.where(
            data['value'] <= 10, '1-10', data['value'])
        data['categories'] = np.where(
            (data['value'] > 10) & (data['value'] <= 50),
            '11-50', data['categories'])
        data['categories'] = np.where(
            (data['value'] > 50) & (data['value'] <= 200),
            '51-200', data['categories'])
        data['categories'] = np.where(
            data['value'] > 200, '> 200', data['categories'])

        # Merge with world map (ensure country names match)
        world = world.merge(data, how='left', left_on='name',
                            right_on='country')

        # check if all countries are there
        for c in data['country']:
            if c not in list(world['name']):
                print(f'{c} missing in world map!')

        world = world.to_crs('ESRI:54042')  # Winkel Trippel Projection

        # Plot map
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        world.plot(column='categories', cmap='cividis_r', legend=True, ax=ax,
                   categorical=True, edgecolor='dimgrey', linewidth=.4,
                   missing_kwds={'color': 'white', 'label': 'No Data'})
        plt.title("global numbers of submission")

        self.plot_end(filename)

    def generation_perc_bar_generic(self, df: pd.DataFrame, question: str,
                                    answers: list, filename: str) -> None:
        '''stacked barchart for single choice questions'''

        df.dropna(subset=question, inplace=True)

        # prepare percentage wise generation group for question
        df_generation = pd.DataFrame(index=self.GENERATIONS)
        df_grouped = df.groupby(['generation', question]).size().unstack()
        df_generation = pd.merge(left=df_generation, right=df_grouped,
                                 how='outer',
                                 left_index=True, right_index=True)
        df_generation = df_generation.loc[self.GENERATIONS]
        # compute percentages
        sums = df_generation.sum(axis=1)
        for c in df_generation.columns:
            df_generation[c] = (df_generation[c]/sums) * 100

        df_generation.fillna(0, inplace=True)

        # plot results
        fig, ax = plt.subplots(figsize=(7, 4.5))

        bottom = np.zeros(len(df_generation))
        x = np.arange(len(df_generation))

        for i, answer in enumerate(answers):
            try:
                ax.bar(x, df_generation[answer],
                       width=0.7,
                       label=self.wrap_text(answer, 25),
                       bottom=bottom, color=self.c_colors[i])
                # annotation
                for j, x_ in enumerate(x):
                    perc = int(round(df_generation[answer].iloc[j], 0))
                    if perc > 7:
                        ax.text(x=x_, y=bottom[j]+perc/2,
                                s=f'{perc}%', ha='center', va='center')
                bottom += df_generation[answer].values
            except KeyError:
                pass

        ax.set_ylim(bottom=0, top=100)
        ax.set_xlim(left=x.min()-0.4, right=x.max()+0.4)

        ax.set_title(self.wrap_text(question, max_length=50))
        ax.set_xlabel("generation")
        ax.set_ylabel("percent %")
        ax.grid(alpha=0.5, axis='y')
        ax.set_xticks(x)

        ticklabels = []
        for s, g in zip(sums, df_generation.index):
            ticklabels.append(f'{g}\nn={int(s)}')
        ax.set_xticklabels(ticklabels, rotation=0)

        ax.legend(bbox_to_anchor=(1.01, 1.0), reverse=True)

        self.plot_end(filename)

    def generation_multichoice_hbars(self, df: pd.DataFrame, question: str,
                                     question_nr: int,
                                     answers: list, filename: str):
        '''multiple choice hbars per possible answer'''

        df = df[df['generation'].isin(self.GENERATIONS)]

        # Step 1: Stack the choice columns to get a long-format dataframe
        choice_columns = [f'{question_nr}_choice_{i}' for i in [1, 2, 3]]
        df_long = df.melt(id_vars=['generation'], value_vars=choice_columns,
                          value_name='choice')
        df_long = df_long.dropna(subset=['choice'])

        # Step 2: Count how many participants from each generation chose each option
        counts = df_long.groupby(['choice', 'generation']).size().unstack(fill_value=0)

        # Step 3: Normalize by total participants per generation to get %
        generation_counts = df['generation'].value_counts()
        percentages = counts.divide(generation_counts, axis=1) * 100

        # Step 4: Reindex to ensure all choices are included
        percentages = percentages.reindex(answers).fillna(0)

        # Step 5: Plotting explicitly with matplotlib
        fig, ax = plt.subplots(figsize=(7, len(answers)*0.55))

        bar_height = 0.15
        generations_order = percentages.columns.tolist()
        y_positions = np.arange(len(answers))

        reversed_y_positions = y_positions[::-1]
        for i, gen in enumerate(generations_order):
            offset = (i - len(generations_order)/2) * bar_height + bar_height/2
            ax.barh(reversed_y_positions + offset, percentages[gen],
                    height=bar_height, label=gen, color=self.c_colors[i])

        # Format plot
        ax.set_yticks(y_positions)
        answers = [self.wrap_text(a, 30) for a in answers]
        ax.set_yticklabels(answers[::-1])
        ax.set_xlabel('percent %')
        ax.set_ylabel('choices')
        ax.set_title(self.wrap_text(question, max_length=60))
        ax.legend(title='generation', bbox_to_anchor=(1.01, 1),
                  loc='upper left', reverse=True)
        ax.grid(axis='x', alpha=0.5)

        self.plot_end(filename)

    def generation_perc_bar_multichoice(self, df: pd.DataFrame, question: str,
                                        question_nr: int,
                                        answers: list, filename: str) -> None:
        '''column chart for multiple choice questions
        DISCONTINUED because statistically questionable'''
        df_generation = pd.DataFrame(index=self.GENERATIONS, columns=answers)

        choice_cols = [f'{question_nr}_choice_{i}' for i in [1, 2, 3]]

        for g in self.GENERATIONS:
            df_g = df[df['generation'] == g]

            # Combine and flatten all selected columns
            combined = pd.concat([df_g[col] for col in choice_cols])
            # Get value counts not including NaN
            value_counts = combined.value_counts(dropna=True)
            # check if all possible answers are there
            value_counts = value_counts.reindex(answers, fill_value=0)
            value_counts = (value_counts / value_counts.sum()) * 100
            df_generation.loc[g, answers] = value_counts.values


        # plot results
        fig, ax = plt.subplots(figsize=(7, 4.5))

        bottom = np.zeros(len(df_generation))
        x = np.arange(len(df_generation))

        for i, answer in enumerate(answers):
            try:
                ax.bar(x, df_generation[answer],
                       width=0.7,
                       label=self.wrap_text(answer, 25),
                       bottom=bottom, color=self.c_colors[i])
                bottom += df_generation[answer]
            except KeyError:
                pass

        ax.set_ylim(bottom=0, top=100)
        ax.set_xlim(left=x.min()-0.4, right=x.max()+0.4)

        ax.set_title(self.wrap_text(question, max_length=50))
        ax.set_xlabel("generation")
        ax.set_ylabel("percent %")
        ax.grid(alpha=0.5, axis='y')
        ax.set_xticks(x)

        ax.set_xticklabels(df_generation.index, rotation=0)
        ax.legend(bbox_to_anchor=(1.01, 1.0), reverse=True)

        self.plot_end(filename)

    def generation_violin_generic(self, df: pd.DataFrame, question: str,
                                  filename: str) -> None:
        '''plots the distribution of answers to numerical questions as violins'''

        df.dropna(subset=question, inplace=True)

        df[question] = pd.to_numeric(df[question], errors='coerce')

        values = []
        for g in self.GENERATIONS:
            try:
                values.append(df[df['generation'] == g][question].dropna().values)
            except KeyError:
                values.append([])

        fig, ax = plt.subplots(figsize=(5, 4))

        parts = ax.violinplot(values, showmeans=False,
                              showmedians=False, showextrema=False, points=20,
                              widths=0.8)

        for pc in parts['bodies']:
            pc.set_facecolor(self.c_colors[0])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        p5 = [np.percentile(v, 5) for v in values]
        q1 = [np.percentile(v, 25) for v in values]
        medians = [np.median(v) for v in values]
        means = [np.mean(v) for v in values]
        q3 = [np.percentile(v, 75) for v in values]
        p95 = [np.percentile(v, 95) for v in values]
        sums = [len(v) for v in values]

        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=50, zorder=3)
        ax.scatter(inds, means, marker='_', color='white', s=80, zorder=3)
        ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, p5, p95, color='k', linestyle='-', lw=1)

        ax.grid(alpha=0.5)

        ax.set_xticks(np.arange(len(self.GENERATIONS))+1)

        ticklabels = []
        for s, g in zip(sums, self.GENERATIONS):
            ticklabels.append(f'{g}\nn={int(s)}')
        ax.set_xticklabels(ticklabels, rotation=0)

        ax.set_title(self.wrap_text(question, max_length=50))
        ax.set_xlabel("generation")
        # ax.set_ylabel("percent %")

        self.plot_end(filename)

    def generation_rating_plot(self, df: pd.DataFrame, columns: list, answers: list,
                               question: str, filename: str):
        '''plot visualizes answers to rating questions over generations'''
        fig = plt.figure(figsize=(10, len(columns)/1.5))

        for i, g in enumerate(self.GENERATIONS):
            df_temp = df.dropna(subset=columns)
            df_temp = df_temp[df_temp['generation'] == g]
            df_temp = df_temp[columns].astype(int, copy=True)
            summary_df = df_temp.apply(lambda x: x.value_counts(normalize=True)).T
            summary_df.fillna(value=0, inplace=True)
            summary_data = summary_df.values*100
            summary_data_cum = summary_data.cumsum(axis=1)

            ax = fig.add_subplot(1, len(self.GENERATIONS), i+1)
            for j in range(len(summary_df.columns)):
                widths = summary_data[:, j]
                starts = summary_data_cum[:, j] - widths
                ax.barh(summary_df.index, widths, left=starts, height=0.8,
                        color=self.c_colors[j])

            ax.set_title(g)
            ax.invert_yaxis()

            ax.set_xlim(left=0, right=100)
            ax.set_xticks([0, 33, 66, 100])
            ax.set_xlabel('percent %')

            if i == 0:
                ax.set_yticks(range(len(summary_df)))
                choices = [self.wrap_text(ch, max_length=40) for ch in summary_df.index]
                ax.set_yticklabels(choices)
            else:
                ax.get_yaxis().set_visible(False)

            ax.grid(alpha=0.5, axis='x')

        # account for questions with "I don't know"
        if min(summary_df.columns) < 0:
            answers.reverse()

        answers = [self.wrap_text(a, max_length=20) for a in answers]
        custom_lines = [Line2D([0], [0], color=self.c_colors[c], lw=4) for c in range(len(answers))]
        ax.legend(custom_lines, answers, bbox_to_anchor=(1.01, 1.0))

        fig.suptitle(self.wrap_text(question, max_length=70))

        self.plot_end(filename)
