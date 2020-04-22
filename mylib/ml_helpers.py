# +
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def get_feature_selected_columns(cut_away_percentage, importances, colnames):
    '''
    importances - feat_importances[]
    cut_away_percentage - (0-1)
    colnames - of df_features
    '''
    
    take_nr_features = len(importances) - int(len(importances) * cut_away_percentage)

    name_importance_pair = list(zip(colnames, importances))

    name_importance_pair.sort(key=lambda tup: tup[1], reverse=True)

    name_importance_pair = name_importance_pair[:take_nr_features]

    important_names = [i[0] for i in name_importance_pair]
    
    return important_names

def get_selected_correlation_features(df, cut_away_percentage, label):
    '''
    df - all_transformations[x]
    cut_away_percentage - (0-1)
    label - label that we want to measure correlation on
    '''
    cor = df.corr()
    
    cor_target = abs(cor[label])
    
    cut_nr = cor_target.shape[0] - int(cor_target.shape[0] * cut_away_percentage)

    cor_target = cor_target.drop(label).sort_values(by=label, ascending=False).index.to_list()[:cut_nr]
    
    return cor_target



def plot_confusion_matrix(y_test, prediction, filename):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, prediction)
    
    fig, ax =plt.subplots(figsize=(10, 10))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.heatmap(cm, cmap=cmap, square=True, annot=True, cbar=True, fmt='g')

    plt.title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("Truth", fontsize=16)
    
    plt.savefig(filename)
    plt.show()

def get_constant_features(df):
    '''
    return list of constant features to be excluded from the data
    '''
    s = df.std()
    mask = s.values == 0
    constants = list(s.index[mask])
    return constants

def feature_correlation(features, filename='feature_corr_matrix.csv', save=False, plot=True):
    '''
    absolute correlation
    '''
    cor = features.corr().abs()
    cor_upper = cor.where(np.tril(np.ones(cor.shape), k=0).astype(np.bool))
    
    if save:
        cor_upper.to_csv(filename, index=False)
        
    if plot:
        import seaborn as sns
        fig, ax =plt.subplots(figsize=(12, 12))
        plt.title("Correlation Plot")
        
        # optional - set the mask to triangle False
        mask = np.array(cor_upper)
        mask[~np.isnan(mask)] = False
        
        if cor_upper.shape[0] < 40:
            annotate = True
        else:
            annotate = False
        sns.heatmap(cor, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax, annot=annotate, cbar=True)
        plt.show()
    return cor_upper

def plot_feature_correlation(df_train, cor_matrix, corr_thresh, export='high_correlated_features.png'):
    '''
    take arg cor_upper from feature_correlation return
    '''
    s = cor_matrix[cor_matrix > corr_thresh].unstack()
    so = s[s.sort_values(kind="quicksort").notnull()]
    
    n_plots = len(so)
    n_plots_x = int(len(so)/3)
    n_plots_y = len(so)-n_plots_x
    place = 1

    fig = plt.figure(figsize=(20,50))

    for index, value in so.items():
        ax = fig.add_subplot(n_plots_x+n_plots_y, n_plots_x, place)
        scatter = ax.scatter(x=df_train[index[0]], y=df_train[index[1]], c=df_train.Cover_Type)
        ax.set_xlabel(index[0])
        ax.set_ylabel(index[1])
        ax.set_title('abs. correlation '+str(round(value, 2)))

        if place == 2:
            legend = ax.legend(*scatter.legend_elements(), loc='upper center', bbox_to_anchor=(0.4, 1.5),
                      ncol=n_plots, fancybox=True, title='classes')
            ax.add_artist(legend)
        place+=1
    plt.savefig(export)
    plt.show()


# +
def get_skewed(df, figsize=(15,25), hspace=0.75, upperBound=0.5, lowerBound=-0.5):
    df = df._get_numeric_data()
    np.seterr(divide='ignore', invalid='ignore')
    fig, ax = plt.subplots(df.shape[1],2,figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    skewed = dict()

    for idx, column in enumerate(df.columns):
        sns.boxplot(x=df[column],ax=ax[idx,0])
        sns.distplot(df[column],ax=ax[idx,1])
        skewed[column] = df[column].skew(skipna=True)
    
    plt.tight_layout()
    unskewed_cols = {key:value for key,value in skewed.items() if (value < upperBound and value > lowerBound)}
    skewed_cols = {key:value for key,value in skewed.items() if (value >= upperBound or value <= lowerBound)}
    

    return skewed_cols, unskewed_cols

def display_outliers(df,func=display):
    df = df._get_numeric_data()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bounds = Q1-IQR*1.5
    upper_bounds = Q3+IQR*1.5
  
    for column in upper_bounds.keys():
        filtered = df[(df[column]<lower_bounds[column])\
            | (df[column]>upper_bounds[column])]
        if filtered.shape[0]>0:
            print(column)
            func(filtered[:5]) # max of 5 
# -


