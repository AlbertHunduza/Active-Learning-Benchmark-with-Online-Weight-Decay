import sys
sys.path.append("core")
sys.path.append(".")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx
from core.evaluation import (
   combine_agents_into_df, 
   average_out_columns
)

# Strategy name corrections
strategy_corrections = {
   'constant': 'static',
   'optimal': 'grid search',
   'predict': 'meta-learning model'
}

# Agent name corrections
name_corrections = {
   "RandomAgent": "Random",
   "Coreset_Greedy": "Coreset",
   "ShannonEntropy": "Entropy",
   "MarginScore": "Margin"
}

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
               width=6, textspace=1, reverse=False, filename=None, labels=False,
               rank_text_size=8, bigtick = 0.2, smalltick = 0.1, space_between_names = 0.25, **kwargs):
   """
   Draws a CD graph, which is used to display the differences in methods'
   performance.
   """
   width = float(width)
   textspace = float(textspace)

   def nth(l, n):
       """Returns only nth element in a list."""
       n = lloc(l, n)
       return [a[n] for a in l]

   def lloc(l, n):
       """List location in list of list structure."""
       if n < 0:
           return len(l[0]) + n
       else:
           return n

   def mxrange(lr):
       """Multiple xranges. Can be used to traverse matrices."""
       if not len(lr):
           yield ()
       else:
           index = lr[0]
           if isinstance(index, int):
               index = [index]
           for a in range(*index):
               for b in mxrange(lr[1:]):
                   yield tuple([a] + list(b))

   sums = avranks
   nnames = names
   ssums = sums

   if lowv is None:
       lowv = min(1, int(math.floor(min(ssums))))
   if highv is None:
       highv = max(len(avranks), int(math.ceil(max(ssums))))

   cline = 0.4
   k = len(sums)
   lines = None
   linesblank = 0
   scalewidth = width - 2 * textspace

   def rankpos(rank):
       if not reverse:
           a = rank - lowv
       else:
           a = highv - rank
       return textspace + scalewidth / (highv - lowv) * a

   distanceh = 0.25
   cline += distanceh

   # Calculate height needed height of an image
   minnotsignificant = max(2 * 0.2, linesblank)
   height = cline + ((k + 1) / 2) * 0.3 + minnotsignificant

   fig = plt.figure(figsize=(width, height))
   fig.set_facecolor('white')
   ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
   ax.set_axis_off()

   hf = 1. / height  # height factor
   wf = 1. / width

   def hfl(l):
       return [a * hf for a in l]

   def wfl(l):
       return [a * wf for a in l]

   # Upper left corner is (0,0).
   ax.plot([0, 1], [0, 1], c="w")
   ax.set_xlim(0, 1)
   ax.set_ylim(1, 0)

   def line(l, color='k', **kwargs):
       """Input is a list of pairs of points."""
       ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

   def text(x, y, s, *args, **kwargs):
       ax.text(wf * x, hf * y, s, *args, **kwargs)

   line([(textspace, cline), (width - textspace, cline)], linewidth=2)

   bigtick = 0.3
   smalltick = 0.15
   linewidth = 2.0
   linewidth_sign = 4.0

   tick = None
   for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
       tick = smalltick
       if a == int(a):
           tick = bigtick
       line([(rankpos(a), cline - tick / 2),
             (rankpos(a), cline)],
            linewidth=2)

   for a in range(lowv, highv + 1):
       text(rankpos(a), cline - tick / 2 - 0.05, str(a),
            ha="center", va="bottom", size=16)

   k = len(ssums)

   def filter_names(name):
       return name

   for i in range(math.ceil(k / 2)):
       chei = cline + minnotsignificant + i * space_between_names
       line([(rankpos(ssums[i]), cline),
             (rankpos(ssums[i]), chei),
             (textspace - 0.1, chei)],
            linewidth=linewidth)
       if labels:
           text(textspace + 0.3, chei - 0.075, format(ssums[i], '.1f'), ha="right", va="center", size=rank_text_size)
       text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

   for i in range(math.ceil(k / 2), k):
       chei = cline + minnotsignificant + (k - i - 1) * space_between_names
       line([(rankpos(ssums[i]), cline),
             (rankpos(ssums[i]), chei),
             (textspace + scalewidth + 0.1, chei)],
            linewidth=linewidth)
       if labels:
           text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.1f'), ha="left", va="center", size=rank_text_size)
       text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
            ha="left", va="center", size=16)

   start = cline + 0.2
   side = -0.02
   height = 0.1

   # Draw no significant lines
   # Get the cliques
   cliques = form_cliques(p_values, nnames)
   i = 1
   achieved_half = False
   print(nnames)
   for clq in cliques:
       if len(clq) == 1:
           continue
       min_idx = np.array(clq).min()
       max_idx = np.array(clq).max()
       if min_idx >= len(nnames) / 2 and achieved_half == False:
           start = cline + 0.25
           achieved_half = True
       line([(rankpos(ssums[min_idx]) - side, start),
             (rankpos(ssums[max_idx]) + side, start)],
            linewidth=linewidth_sign)
       start += height

def form_cliques(p_values, nnames):
   """
   This method forms the cliques
   """
   m = len(nnames)
   g_data = np.zeros((m, m), dtype=np.int64)
   for p in p_values:
       if p[3] == False:
           i = np.where(nnames == p[0])[0][0]
           j = np.where(nnames == p[1])[0][0]
           min_i = min(i, j)
           max_j = max(i, j)
           g_data[min_i, max_j] = 1

   g = networkx.Graph(g_data)
   return networkx.find_cliques(g)

def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, width=9.0, file=None):
   """
   Draws the critical difference diagram given the list of pairwise classifiers that are
   significant or not
   """
   p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

   print(average_ranks)
   for p in p_values:
       print(p)

   graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
               cd=None, reverse=True, width=width, textspace=1.5, labels=labels)

   font = {'size': 22}
   if title:
       plt.title(title, fontdict=font, y=0.9, x=0.5)

   if file is not None:
       plt.savefig(file, bbox_inches='tight')
   else:
       plt.show()

def wilcoxon_holm(alpha=0.05, df_perf=None):
   """
   Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
   to reject the null's hypothesis
   """
   print(pd.unique(df_perf['classifier_name']))
   df_counts = pd.DataFrame({'count': df_perf.groupby(['classifier_name']).size()}).reset_index()
   max_nb_datasets = df_counts['count'].max()
   classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['classifier_name'])
   
   friedman_p_value = friedmanchisquare(*(
       np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
       for c in classifiers))[1]
   if friedman_p_value >= alpha:
       print('the null hypothesis over the entire classifiers cannot be rejected')
       
   m = len(classifiers)
   p_values = []
   for i in range(m - 1):
       classifier_1 = classifiers[i]
       perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy'], dtype=np.float64)
       
       for j in range(i + 1, m):
           classifier_2 = classifiers[j]
           perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]['accuracy'], dtype=np.float64)
           p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
           p_values.append((classifier_1, classifier_2, p_value, False))
           
   k = len(p_values)
   p_values.sort(key=operator.itemgetter(2))

   for i in range(k):
       new_alpha = float(alpha / (k - i))
       if p_values[i][2] <= new_alpha:
           p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
       else:
           break

   sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)].sort_values(['classifier_name', 'dataset_name'])
   rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)
   df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=np.unique(sorted_df_perf['dataset_name']))
   average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
   
   return p_values, average_ranks, max_nb_datasets

def prepare_df_for_single(df, agent):
   """Prepare data for single dataset analysis"""
   # Filter for specific agent
   agent_original = {v: k for k, v in name_corrections.items()}.get(agent, agent)
   df = df[df['agent'].str.startswith(f"{agent_original}_")]
   
   # Extract strategy name and apply corrections
   df['classifier_name'] = df['agent'].apply(lambda x: x.split('_')[-1])
   df['classifier_name'] = df['classifier_name'].replace(strategy_corrections)
   df['dataset_name'] = df['trial'].astype(str)
   df['accuracy'] = df['auc']
   
   return df[['classifier_name', 'dataset_name', 'accuracy']]

def analyze_single_dataset(dataset, agent, query_size):
   """Generate CD diagram for single dataset-agent combination"""
   output_dir = "results/single_dataset_cdd"
   os.makedirs(output_dir, exist_ok=True)
   
   # Get data
   df = combine_agents_into_df(dataset=dataset, query_size=query_size, 
                             max_loaded_runs=50, include_oracle=False)
   df = average_out_columns(df, ["iteration"])
   df = prepare_df_for_single(df, agent)
   
   # Create filename and title
   filename = f"{dataset}_{agent}_q{query_size}.png"
   title = f"Strategy Comparison for {agent} on {dataset} (Query Size {query_size})"
   filepath = os.path.join(output_dir, filename)
   
   # Generate CD diagram
   draw_cd_diagram(df, title=title, file=filepath)
   
   return df

if __name__ == '__main__':
   # Configuration
   datasets = ["USPS", "DNA", "Splice", "TopV2"]
   agents = ["BALD", "Badge", "Coreset", "Margin", "Random", "Entropy", "TypiClust"]
   query_sizes = ["20", "50"]
   
   # Generate CD diagrams for all combinations
   for dataset in datasets:
       for agent in agents:
           print(f"\nProcessing {dataset} - {agent}")
           for query_size in query_sizes:
               try:
                   df = analyze_single_dataset(dataset, agent, query_size)
                   print(f"Generated CD diagram for {dataset} - {agent} - q{query_size}")
               except Exception as e:
                   print(f"Error processing {dataset} - {agent} - q{query_size}: {str(e)}")