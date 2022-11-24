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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog as fd
from tkinter import *
from tkinter.messagebox import showinfo
root = tk.Tk()

# config the root window
root.resizable(False, False)
root.title('Air Quality Index Analysis')

w = 800
h = 450

ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

root.geometry('%dx%d+%d+%d' % (w, h, x, y))

bg = PhotoImage(file="img/bg.png")

# Create Canvas
canvas1 = tk.Canvas(root, width = 800, height = 450,  relief = 'raised', highlightthickness=0)
canvas1.pack(expand = YES, fill = BOTH)
# canvas1.grid(row=0,column=1)

# Display image
canvas1.create_image(0, 0, image=bg, anchor="nw")

datasetpath = tk.StringVar()
inputaqi = tk.StringVar()
r2_value = tk.StringVar()
r2_value1 = tk.StringVar()

# df1=pd.read_csv(fd.askopenfilename())
# df1 = pd.read_csv('data/city_day.csv')
# df1.head()
# df1['City'].unique()
# df = df1[df1['City'] == 'Delhi']

def callback():
    dataset = fd.askopenfilename()
    Dataset_path.insert(END, dataset)

def pre_process():
    df = pd.read_csv(datasetpath.get())
    # df = df1[df1['City'] == 'Delhi']
    # df1['City'].unique()
    df.head()
    df.shape
    print(df.head())
    print(df.info())

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
    f, ax_ = plt.subplots(1, 3, figsize=(14, 14))
    # plt.title("Most Polluted Cities", y=1.3, fontsize = 16)

    bar1 = sns.barplot(x=most_polluted.AQI,
                       y=most_polluted.index,
                       palette='Reds_r',
                       ax=ax_[0])

    bar1 = sns.barplot(x=most_polluted.PM10,
                       y=most_polluted.index,
                       palette='RdBu',
                       ax=ax_[1])

    bar1 = sns.barplot(x=most_polluted.CO,
                       y=most_polluted.index,
                       palette='RdBu',
                       ax=ax_[2])

    titles = 'Most Polluted Cities'
    for i in range(3):
        ax_[i].set_ylabel('')
        ax_[i].set_yticklabels(labels=ax_[i].get_yticklabels(), fontsize=4);
        ax_[i].set_title(titles)
        f.tight_layout()

        plt.show()

    cols = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2']

    cmap = plt.get_cmap('Spectral')
    color = [cmap(i) for i in np.linspace(0, 1, 8)]
    explode = [0.2, 0, 0, 0, 0, 0, 0, 0]

    for col in cols:

        plt.figure(figsize=(2.8, 1.8))
        plt.title("Percentage of " + col + " In Most Polluted Cities")
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
    r2.insert(END, format(
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
    r22.insert(END, format(
        r2_score((y_test), (pred))))

def predictAQ():
    df = pd.read_csv(datasetpath.get())
    # df[['City', 'AQI']].groupby('City').mean().sort_values('AQI').plot(kind='bar', cmap='Blues_r', figsize=(8, 8))
    # plt.title('Average AQI in last 5 years')
    # plt.show()
    final_df = df[['AQI', 'AQI_Bucket']].copy()
    print(final_df)

    print(final_df['AQI_Bucket'].unique())
    final_df = df.dropna()
    final_df['AQI_Bucket'] = final_df['AQI_Bucket'].map({'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4, 'Severe': 5}).astype(int)  # mapping numbers
    print(final_df.head())
    # final_df = df.fillna(0)

    X = final_df[['AQI']]
    y = final_df[['AQI_Bucket']]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # X_test = X_test.fillna(X_train.mean())
    clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # user input
    AQI = float(inputaqi.get())

    output = clf.predict([[AQI]])
    from operator import itemgetter
    ind_pos = [0]
    pred_value=itemgetter(*ind_pos)(output)
    print(output)
    if pred_value == 0:
        messagebox.showinfo("Air Quality Prediction", 'Good')
    elif pred_value == 1:
        messagebox.showinfo("Air Quality Prediction", 'Satisfactory')
    elif pred_value == 2:
        messagebox.showinfo("Air Quality Prediction", 'Moderate')
    elif pred_value == 3:
        messagebox.showinfo("Air Quality Prediction", 'Poor')
    elif pred_value == 4:
        messagebox.showinfo("Air Quality Prediction", 'Very Poor')
    elif pred_value == 5:
        messagebox.showinfo("Air Quality Prediction", 'Severe')


lblHead = tk.Label(root, text="Air Quality Index Analysis", font=('Arial', 25))
lblHead.config(font=('helvetica', 14))
canvas1.create_window(400, 30, anchor="center", window=lblHead)

Dataset_path = Entry(root, textvariable=datasetpath ,width= 42)
Dataset_path1 = canvas1.create_window(450, 90, window=Dataset_path)

LoadButton=Button(root, text='Load Dataset', command=callback, bg="#fff000")
LoadDatasetButton1 = canvas1.create_window(230, 80, anchor="nw", window=LoadButton)

PreProcess = Button(root, text="PreProcess", command=pre_process, bg="#fff000")
PreProcess1 = canvas1.create_window(370, 110, anchor="nw", window=PreProcess)

Linear_Regression = Button(root, text="LinearRegression", command=linear_regression, bg="#fff000")
Linear_Regression1 = canvas1.create_window(350, 150, anchor="nw", window=Linear_Regression)

r2lbl = tk.Label(root, text="Train R2 Score")
canvas1.create_window(270, 200, window=r2lbl)
r2 = Entry(root, textvariable=r2_value, width=36)
r21 = canvas1.create_window(430, 200, window=r2)


r2lbl1 = tk.Label(root, text="Test R2 Score")
canvas1.create_window(270, 230, window=r2lbl1)
r22 = Entry(root, textvariable=r2_value1, width=36)
r221 = canvas1.create_window(430, 230, window=r22)

# Label Creation
lbl = tk.Label(root, text="Enter Air Quality Index")
canvas1.create_window(400, 310, anchor="center", window=lbl)

input_aqi = Entry(root, textvariable=inputaqi ,width= 42)
input_aqi1 = canvas1.create_window(400, 340, window=input_aqi)

Predict = Button(root, text="Predict", command=predictAQ, bg="#fff000", fg="#000000")
Predict1 = canvas1.create_window(370, 370, anchor="nw", window=Predict)


root.mainloop()