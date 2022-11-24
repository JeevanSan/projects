# import Button as Button
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sm as sm
from matplotlib import rcParams

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from PIL import ImageTk, Image
import itertools
import statsmodels.api as sm
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog as fd
from tkinter import *
from tkinter.messagebox import showinfo
root = tk.Tk()
top = Toplevel()

# ---- login window ----
def login():
    #getting form data
    uname=username.get()
    pwd=password.get()
    #applying empty validation
    if uname=='' or pwd=='':
        message.set("fill the empty field!!!")
    else:
      if uname=="jeevan" and pwd=="jeevan123":
       message.set("Login success")
       root.deiconify()  # Unhides the root window
       top.destroy()  # Removes the toplevel window
      else:
       message.set("Wrong username or password!!!")

global login_screen

w = 500
h = 300

ws = top.winfo_screenwidth()
hs = top.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
# top.configure(background='white')
top.geometry('%dx%d+%d+%d' % (w, h, x, y))
top.title("Login Form")
# #setting height and width of screen
# top.geometry("500x300")
#declaring variable
global  message;
global username
global password
username = StringVar()
password = StringVar()
message=StringVar()
#Creating layout of login form
# Label(top,width="300", text="Please enter details below", bg="orange",fg="white").pack()

Label(top, text="User Login ", font=('arial', 16)).place(x=190,y=40)

#Username Label
Label(top, text="Username * ").place(x=140,y=80)
#Username textbox
Entry(top, textvariable=username).place(x=210,y=82)
#Password Label
Label(top, text="Password * ").place(x=140,y=120)
#Password textbox
Entry(top, textvariable=password ,show="*").place(x=210,y=122)
#Label for displaying login status[success/failed]
Label(top, text="",textvariable=message).place(x=210,y=160)
#Login button
Button(top, text="Login", width=10, height=1, bg="blue", fg="white" ,command=login).place(x=210,y=190)


# end login

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
pm2_5 = tk.StringVar()
pm10 = tk.StringVar()
no = tk.StringVar()
no2 = tk.StringVar()
nox = tk.StringVar()
co = tk.StringVar()
so2 = tk.StringVar()
o3 = tk.StringVar()
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
    print('---- X Train ----')
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
    # importing Randomforest
    # from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    df = df[df['City'] == 'Ahmedabad']
    df.fillna(0, inplace=True)
    aq_df = df[['PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3']].copy()
    print(aq_df)
    # creating model
    RFR = RandomForestRegressor()
    target = df['AQI']

    # Fitting the model
    RFR.fit(aq_df, target)

    # calculating the score
    print(RFR.score(aq_df, target) * 100)

    # predicting the model with other values (testing the data)
    print(RFR.predict([[22.9,120.7,11.79,15.81,10.1,0.5,10.1,46.04]]))
    # 'PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3'

    AQI_Value = RFR.predict([[
        float(pm2_5.get()), float(pm10.get()), float(no.get()), float(no2.get()),
        float(nox.get()), float(co.get()), float(so2.get()), float(o3.get())
    ]])
    # messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value))
    # messagebox.showinfo("Air Quality Prediction", RFR.predict([[
    #     float(pm2_5.get()), float(pm10.get()), float(no.get()), float(no2.get()),
    #     float(nox.get()), float(co.get()), float(so2.get()), float(o3.get())
    # ]]))
    predict_input = RFR.predict([[22.9, 120.7, 11.79, 15.81, 10.1, 0.5, 10.1, 46.04]])

    df1 = pd.read_csv(datasetpath.get())
    final_df = df1[['AQI', 'AQI_Bucket']].copy()
    print(final_df)

    print(final_df['AQI_Bucket'].unique())
    final_df = df1.dropna()
    final_df['AQI_Bucket'] = final_df['AQI_Bucket'].map({'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4, 'Severe': 5}).astype(int)  # mapping numbers
    print(final_df.head())
    # final_df = df.fillna(0)

    X = final_df[['AQI']]
    y = final_df[['AQI_Bucket']]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_test = X_test.fillna(X_train.mean())
    clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # user input
    # AQI = float(inputaqi.get())
    AQI = float(AQI_Value)
    output = clf.predict([[AQI]])
    from operator import itemgetter
    ind_pos = [0]
    pred_value=itemgetter(*ind_pos)(output)
    print(output)
    if pred_value == 0:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Good')
    elif pred_value == 1:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Satisfactory')
    elif pred_value == 2:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Moderate')
    elif pred_value == 3:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Poor')
    elif pred_value == 4:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Very Poor')
    elif pred_value == 5:
        messagebox.showinfo("Air Quality Prediction", 'Predicted AQI: ' + str(AQI_Value) + ': Severe')


# def openNewWindow(result):
#     # Toplevel object which will
#     # be treated as a new window
#     newWindow = Toplevel(root)
#     newWindow.title("New Window")
#     newWindow.geometry("700x500")
#
#     frame = Frame(newWindow, width=600, height=400)
#     frame.pack()
#     frame.place(anchor='center', relx=0.5, rely=0.5)
#
#     # Create an object of tkinter ImageTk
#     img = ImageTk.PhotoImage(Image.open("img/bg1.png"))
#
#     # Create a Label Widget to display the text or Image
#     label = Label(frame, image=img)
#     label.pack()
#     Label(newWindow,
#           text=result).pack()



def FpredictAQ():
    # openNewWindow("New Window")
    dd = pd.read_csv(datasetpath.get())
    dd_col = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    print(dd.isnull().sum())
    for i in dd_col:
        a = dd[i].mean()
        dd[i].replace(np.nan, a, inplace=True)

    print(dd.isnull().sum())
    Ahemdabad = dd.loc[dd['City'] == 'Ahmedabad']
    Ahemdabad.head(10)
    print(Ahemdabad.shape)
    Ahemdabad['Date'] = pd.to_datetime(Ahemdabad['Date'])

    print(Ahemdabad.columns)
    c = ['City', 'Benzene', 'Toluene', 'Xylene', 'AQI_Bucket']
    # c = ['Benzene', 'Toluene', 'Xylene', 'AQI_Bucket']
    Ahemdabad.drop(c, axis=1, inplace=True)
    Ahemdabad = Ahemdabad.sort_values('Date')
    print(Ahemdabad.tail(10))
    Ahemdabad = Ahemdabad.reset_index()
    Ahemdabad = Ahemdabad.set_index('Date')
    Ahemdabad.index
    print(Ahemdabad.tail(10))

    PM25 = Ahemdabad['PM2.5'].resample('2W').mean()
    PM25.plot(figsize=(15, 6))
    plt.show()
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 15
    decomposition = sm.tsa.seasonal_decompose(PM25, model='additive')
    fig1 = decomposition.plot()
    plt.show()

    # SAMIRA Model for PM2.5
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    mod = sm.tsa.statespace.SARIMAX(PM25,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)


    y_forecasted = pred.predicted_mean
    y_truth = PM25['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = PM25.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('PM2.5 Concentration')
    plt.legend()
    plt.show()

    # PM10
    PM10 = Ahemdabad['PM10'].resample('2W').mean()
    # SAMIRA Model for PM10
    p = d = q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    mod = sm.tsa.statespace.SARIMAX(PM10,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    y_forecasted = pred.predicted_mean
    y_truth = PM10['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = PM10.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('PM10 Concentration')
    plt.legend()
    plt.show()


    # aqi forecast
    AQI = Ahemdabad['AQI'].resample('2W').mean()
    # SAMIRA Model for AQI
    p = d = q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    mod = sm.tsa.statespace.SARIMAX(AQI,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 1, 48),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    y_forecasted = pred.predicted_mean
    y_truth = AQI['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = AQI.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('AQI Concentration')
    plt.legend()
    plt.show()

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
pm2_5_lbl = tk.Label(root, text="PM2.5")
canvas1.create_window(150, 310, anchor="center", window=pm2_5_lbl)

pm10_lbl = tk.Label(root, text="PM10")
canvas1.create_window(220, 310, anchor="center", window=pm10_lbl)

no_lbl = tk.Label(root, text="NO")
canvas1.create_window(290, 310, anchor="center", window=no_lbl)

no2_lbl = tk.Label(root, text="NO2")
canvas1.create_window(360, 310, anchor="center", window=no2_lbl)

nox_lbl = tk.Label(root, text="NOx")
canvas1.create_window(430, 310, anchor="center", window=nox_lbl)

co_lbl = tk.Label(root, text="CO")
canvas1.create_window(500, 310, anchor="center", window=co_lbl)

so2_lbl = tk.Label(root, text="SO2")
canvas1.create_window(570, 310, anchor="center", window=so2_lbl)

o3_lbl = tk.Label(root, text="O3")
canvas1.create_window(640, 310, anchor="center", window=o3_lbl)

pm2_5 = Entry(root, textvariable=pm2_5 ,width= 10)
canvas1.create_window(150, 340, window=pm2_5)

pm10 = Entry(root, textvariable=pm10 ,width= 10)
canvas1.create_window(220, 340, window=pm10)

no = Entry(root, textvariable=no ,width= 10)
canvas1.create_window(290, 340, window=no)

no2 = Entry(root, textvariable=no2 ,width= 10)
canvas1.create_window(360, 340, window=no2)

nox = Entry(root, textvariable=nox ,width= 10)
canvas1.create_window(430, 340, window=nox)

co = Entry(root, textvariable=co ,width= 10)
canvas1.create_window(500, 340, window=co)

so2 = Entry(root, textvariable=so2 ,width= 10)
canvas1.create_window(570, 340, window=so2)

o3 = Entry(root, textvariable=o3 ,width= 10)
canvas1.create_window(640, 340, window=o3)

Predict = Button(root, text="Predict", command=predictAQ, bg="#fff000", fg="#000000")
Predict1 = canvas1.create_window(370, 370, anchor="nw", window=Predict)

F_Predict = Button(root, text="Future Predict", command=FpredictAQ, bg="#fff000", fg="#000000")
canvas1.create_window(370, 400, anchor="nw", window=F_Predict)

root.withdraw()
root.mainloop()