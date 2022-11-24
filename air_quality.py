# import Button as Button
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import chart_studio.plotly as py
# import plotly.graph_objs as go
from matplotlib import rcParams
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import cufflinks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import *
from tkinter.messagebox import showinfo
root = tk.Tk()

# config the root window
root.geometry('1200x800')
root.resizable(False, False)
root.title('Combobox Widget')
datasetpath = tk.StringVar()
Dataset_path = Entry(root, textvariable=datasetpath ,width= 42)
Dataset_path.pack(padx=5, pady=5)

# df1=pd.read_csv(fd.askopenfilename())
# df1 = pd.read_csv('data/city_day.csv')
# df1.head()
# df1['City'].unique()
# df = df1[df1['City'] == 'Delhi']

def callback():
    dataset = fd.askopenfilename()
    # df1 = pd.read_csv(dataset)
    Dataset_path.insert(END, dataset)
    # print(df1)
    # print(name)

# def show_data():
    # df = pd.read_csv(datasetpath.get())
    # # df = df1[df1['City']]
    # df.head()
    # df.shape
    # print(df.head())

def pre_process():
    df = pd.read_csv(datasetpath.get())
    # df = df1[df1['City'] == 'Delhi']
    # df1['City'].unique()
    df.head()
    df.shape
    print(df.head())
    print(df.info())

    # # show heatmap
    # plt.figure(figsize=(12, 8))
    #
    # mask = np.triu(df.corr(method='pearson'))
    # sns.heatmap(df.corr(method='pearson'),
    #             annot=True, fmt='0.1f',
    #             mask=mask,
    #             robust=True,
    #             cmap='pink')
    # plt.title('Correlation Analysis', fontsize=18);
    # # plt.savefig('img/correlation_analysis.png')
    # plt.show()
    # print(df.describe())
    # sns.boxplot(data=df)
    # plt.show()
    # plt.figure(figsize=(8, 4), dpi=200)
    # palette = {'Good': "g", 'Poor': "C0", 'Very Poor': "C1", 'Severe': "r", "Moderate": 'b', "Satisfactory": 'y'}
    # sns.scatterplot(x='AQI', y='PM2.5', data=df, hue='AQI_Bucket', palette=palette, ci=None)
    # plt.show()
    #
    most_polluted = df[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by='AQI',ascending=False)
    print(most_polluted)

    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    rcParams['figure.dpi'] = 300

    rcParams['figure.autolayout'] = True

    rcParams['font.style'] = 'normal'
    rcParams['font.size'] = 3

    rcParams['lines.linewidth'] = 0.7

    rcParams['xtick.labelsize'] = 4
    rcParams['ytick.labelsize'] = 4

    plt.style.use('seaborn-whitegrid')
    f, ax_ = plt.subplots(1, 3, figsize=(15, 15))

    bar1 = sns.barplot(x=most_polluted.AQI,
                       y=most_polluted.index,
                       palette='RdBu',
                       ax=ax_[0]);

    bar1 = sns.barplot(x=most_polluted.PM10,
                       y=most_polluted.index,
                       palette='RdBu',
                       ax=ax_[1]);

    bar1 = sns.barplot(x=most_polluted.CO,
                       y=most_polluted.index,
                       palette='RdBu',
                       ax=ax_[2]);

    titles = ['AirQualityIndex', 'ParticulateMatter10', 'CO']
    for i in range(3):
        ax_[i].set_ylabel('')
        ax_[i].set_yticklabels(labels=ax_[i].get_yticklabels(), fontsize=14);
        ax_[i].set_title(titles[i])
        # f.tight_layout()
        plt.show()

    cols = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2']

    cmap = plt.get_cmap('Spectral')
    color = [cmap(i) for i in np.linspace(0, 1, 8)]
    explode = [0.2, 0, 0, 0, 0, 0, 0, 0]

    for col in cols:
        plt.figure(figsize=(2.8, 1.8))

        '''grouping above columns by cities and 
        taking 8 cities which have the highest sum'''

        x = df.groupby('City')[col].sum().sort_values(ascending=False)
        x.reset_index('City')
        x[:8].plot.pie(shadow=True, autopct='%1.1f%%',
                       colors=color, explode=explode,
                       wedgeprops={'edgecolor': 'black', 'linewidth': 0.3}
                       )
        plt.show()

def linear_regression():
    df = pd.read_csv(datasetpath.get())
    df = df.drop(columns=['City', 'Date'])
    # dataset
    df['AQI'] = np.log(df['AQI'])
    df.fillna(-99999, inplace=True)
    # print(df)
    # print(df.isna().sum() / df.shape[0])
    # # Drop all rows where target value, aka AQI index is null
    # df = df[df['AQI'].isna()==False]
    # print(df)
    #
    # # Now calculating the null values
    # print(df.isna().sum()/df.shape[0])

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[[i for i in df.columns if i not in ["AQI","AQI_Bucket"]]],
        df["AQI"],
        test_size=0.2,
        random_state=100
        )

    print(X_train)

    ss = StandardScaler()
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(X_train)

    # Fitting the model with Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    # print(LinearRegression())

    pred = model_lr.predict(X_train)

    print('train mse: {}'.format(
        mean_squared_error((y_train), (pred))))
    print('train rmse: {}'.format(
        mean_squared_error((y_train), (pred), squared=False)))
    print('train r2: {}'.format(
        r2_score((y_train), (pred))))
    print()

    # make predictions for test set
    pred = model_lr.predict(X_test)

    # determine mse, rmse and r2
    print('test mse: {}'.format(
        mean_squared_error((y_test), (pred))))
    print('test rmse: {}'.format(
        mean_squared_error((y_test), (pred), squared=False)))
    print('test r2: {}'.format(
        r2_score((y_test), (pred))))

frame = Frame(root)
frame.pack()


# vlist = ["Delhi", "Bengaluru", "Ahmedabad", "Chennai", "Hyderabad", "Kolkata", "Mumbai", "Jaipur"]
# parameterList = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
# yearList = ["2015", "2016", "2017", "2018", "2019", "2020"]



LoadButton=Button(root, text='Click to Open File', command=callback)
LoadButton.pack(padx=5, pady=5)

# Combo = ttk.Combobox(root, values=vlist)
# Combo.set("Pick an Option")
# Combo.pack(padx=5, pady=5)
#
# Para_Combo = ttk.Combobox(root, values=parameterList)
# Para_Combo.set("Pick an Option")
# Para_Combo.pack(padx=5, pady=5)
#
# Year_Combo = ttk.Combobox(root, values=yearList)
# Year_Combo.set("Pick an Option")
# Year_Combo.pack(padx=5, pady=5)
#
# ShowData = Button(root, text="Submit", command=show_data)
# ShowData.pack(padx=5, pady=5)
#
PreProcess = Button(root, text="PreProcess", command=pre_process)
PreProcess.pack(padx=5, pady=5)

Linear_Regression = Button(root, text="LinearRegression", command=linear_regression)
Linear_Regression.pack(padx=5, pady=5)

root.mainloop()