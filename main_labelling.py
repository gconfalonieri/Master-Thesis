import toml
import experiments
import labelling.set_labels
import statistics.utilities
import pandas as pd

config = toml.load('config.toml')
experiments.utilities.fix_seeds_unique()

df_complete = pd.read_csv('datasets/results/answers_complete_all_info.csv')

df_labelled = labelling.set_labels.get_df_answers_labelled(df_complete, 'TIMES_ONLY_V2')
df_label_statistics = labelling.set_labels.get_df_label_statistics(df_labelled, 'MEDIA_NAME')

df_labelled.to_csv('datasets/results/labels/labelled_dataset_v2.csv', index=False)
df_label_statistics.to_csv('datasets/results/labels/label_statistics_dataset_v2.csv', index=False)

statistics.plots.get_total_labels_pie_plot(df_label_statistics, 'PROPOSAL 1.1 - VARIANT 2', 'plots/')
statistics.plots.get_labels_pie_plot(df_label_statistics, 'MEDIA_NAME', 'plots/media_name/')