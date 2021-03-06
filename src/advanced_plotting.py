import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import glob as glob
import numpy as np
import warnings
import math
from collections import Counter
import scipy.stats as ss


def nightingale_rose_plot(num_col,cat_col1,cat_col2,df):
    """
    Use cases:
    This chart is widely used when showing distribution over time. 
    It is often used to represent wind speed and directions.
    """
    
    import plotly.express as px
    #setting the parameters of the chart
    fig = px.bar_polar(df, r=num_col, theta=cat_col1,  #r is the values, theta= data you wish to compare
                       color=cat_col2, template="plotly_dark")  #color is the value of stacked columns 

    #adding title, circular grid shape and labels
    fig.update_layout(
        title=f'Comparison of {num_col} against {cat_col1} and {cat_col2}',
        template=None,
        polar = dict(gridshape='circular',bgcolor='lightgray',
            radialaxis = dict(range=[0, 50], ticks='')  #setting the scale
        ))
    fig.show()
    
def sankey_plot(cat_col,df):
    """
    This type of visualization can be used to describe the flow of an entity from source to end.
    """
    
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    distinct_cat_list = np.array(df[cat_col].value_counts().index)  #total nodes involved in the graph
    target_label = f"Grand Total {cat_col}"
    distinct_cat_list = np.append(distinct_cat_list,target_label)
    #creating a sankey diagram using plotly
    fig = go.Figure(data=[go.Sankey(       
        node = dict(            #editing properties of the node
          thickness = 15,
          line = dict(color = "black"),
          label = distinct_cat_list, #total nodes
        ),
        #editing properties of the connecting link
        link = dict(               
            source = np.arange(len(distinct_cat_list)),  #source nodes
            target = np.ones(len(distinct_cat_list),dtype=np.int)*(len(distinct_cat_list)),   #target node
            value = np.array(df['outcome_type'].value_counts()),  #value of the links
            color = '#eee0e5'
      ))])

    #setting figure title and font style
    fig.update_layout(title_text=f"Category distribution of {cat_col}", font=dict(size = 12, color = 'maroon'),paper_bgcolor='white')
    fig.show()

def pairplot_num(num_cols,df,title):
    # Single line to create pairplot
    g = sns.pairplot(df[num_cols])
    g.fig.suptitle(title, y=1.08);
    
def pairplot_num_onecat(num_cols,cat_col,df,title):
    # Single line to create pairplot
    num_cols.append(cat_col)
    g = sns.pairplot(df[num_cols],hue=cat_col)
    g.fig.suptitle(title, y=1.08);

def swarmplots_onenum_onecat(num_col,cat_col,context_col,df,title,size):
    max_val = df[num_col].max()
    max_val_idx = df[(df[num_col] == max_val)][context_col].values[0]
    g = sns.boxplot(y = cat_col,
                  x = num_col, 
                  data = df, whis=np.inf)
    g = sns.swarmplot(y = cat_col,
                  x = num_col, 
                  data = df,
                  # Decrease the size of the points to avoid crowding 
                  size = 7,color='black')
    # remove the top and right line in graph
    sns.despine()
    g.set_title(title, y=1.08);
    # Annotate. xy for coordinate. max_wage is x and 0 is y. In this plot y ranges from 0 to 7 for each level
    # xytext for coordinates of where I want to put my text
    plt.annotate(s = max_val_idx,
                 xy = (max_val,0),
                 xytext = (500,1), 
                 # Shrink the arrow to avoid occlusion
                 #arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},
                 backgroundcolor = 'white')
    g.figure.set_size_inches(size*2,size)
    plt.show()
    
def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted
    
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                          return_results = False, **kwargs):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases
    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param nominal_columns: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])
                        else:
                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), cmap='coolwarm')
        plt.show()
    if return_results:
        return corr