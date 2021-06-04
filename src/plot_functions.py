import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import glob as glob
import numpy as np
import warnings
from scipy.stats import ttest_ind,f_oneway

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_count(feature, title, df, size=1):
    fig,ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show() 

def plot_box(num_feature,cat_feature,title,df,size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.boxplot(y=df[num_feature], x=df[cat_feature],orient="v")
    g.set_title("Distribution of {} with respect to {} {}".format(num_feature,cat_feature,title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

def barplot_count_multicat(cat_cols_list,df,size=1,subplots=False,sharex=True):
    if len(cat_cols_list)==1:
        plot_count(cat_cols_list[0], f"{cat_cols_list[0]}", df, size=size)
    else:
        fig, ax = plt.subplots(figsize=(size*4,4))
        title="_vs_".join(cat_cols_list)
        ax.set(ylabel='Count')
        grp_unstaked = df.groupby(cat_cols_list).size().unstack()
        grp_unstaked.plot(ax=ax,kind="bar",title=title,subplots=False,sharex=True,grid=True,sort_columns=True)
        plt.legend(loc='best')
        plt.show()
    
def barplot_pct_multicat(cat_cols_list,df,size=1,subplots=False,sharex=True):
    if len(cat_cols_list)==1:
        plot_count(cat_cols_list[0], f"{cat_cols_list[0]}", df, size=size)
    else:
        fig, ax = plt.subplots(figsize=(size*4,4))
        ax.set(ylabel='Percentage')
        title="_vs_".join(cat_cols_list)
        grp_unstaked_pct = df.groupby(cat_cols_list).size().unstack()*100/df.shape[0]
        grp_unstaked_pct.plot(ax=ax,kind="bar",title=title,subplots=False,sharex=True,grid=True,sort_columns=True)
        plt.legend(loc='best')
        plt.show()
    
def plot_box_category(num_feature,cat_feature,sub_cat_feature,title,df,size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.boxplot(y=num_feature, x=cat_feature, hue=sub_cat_feature, palette="pastel",orient="v",data=df)
    g.set_title("Distribution of {} with respect to {} and {} {}".format(num_feature,cat_feature,sub_cat_feature,title))
    sns.set(style="ticks", palette="pastel")
    sns.despine(offset=10, trim=True)
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
#         ax.text(p.get_x()+p.get_width()/2.,
#                 height + 3,
#                 '{:1.2f}%'.format(100*height/total),
#                 ha="center") 
    plt.show() 

def plot_regplot(x_var, y_var, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.jointplot(x=x_var, y=y_var, data=df, kind='reg')
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    #plt.show()

def kdeplot_2num_1cat_pdf(data,num_varx,num_vary,cat_var,title,shade):
    num_cats = data[cat_var].nunique()
    cmap_cols=["Blues","Reds"]
    for i,cat in enumerate(data[cat_var].unique()):
        df=data.loc[data[cat_var] == cat]
        ax=sns.kdeplot(df[num_varx], df[num_vary],cmap=cmap_cols[i],shade=shade, shade_lowest=False,legend=True,label=cat)
    plt.title('{} vs {} for categories {}: {}'.format(num_varx,num_vary,cat_var,title))
    plt.legend(loc='best')
    plt.show()

def mape(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    inx = np.where(y_test!=0)
    data_perc_size = 100*inx[0].shape[0]/y_test.shape[0]
    y_test = y_test[inx[0]]
    y_pred = y_pred[inx[0]]
    mape = np.mean(np.abs((y_test - y_pred) / y_test))*100
    print("MAPE: {}% based on {}% non-zero test data.".format(np.around(mape,2),np.around(data_perc_size,2)))

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,
                            title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # print(unique_labels(y_true, y_pred))
    classes = classes[unique_labels(y_true, y_pred).astype(int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    print(np.arange(cm.shape[1]))
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.autoscale()
    return ax


# np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test_cls.ravel(), (y_test_cls_pred>0.5).ravel(), classes=np.array([0,1]),
#                         title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test_cls.ravel(), (y_test_cls_pred>0.5).ravel(), classes=np.array([0,1]), normalize=True,
#                         title='Normalized confusion matrix')

#plt.show()