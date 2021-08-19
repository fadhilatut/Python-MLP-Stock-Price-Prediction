import matplotlib
from matplotlib import colors
import numpy as np
import pandas as pd
import tkinter as tk
import os
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from tkinter import*
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from pandastable import Table
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#Main Window
root = Tk()
root.geometry("380x230")
root.resizable(1,1)
#f1 = LabelFrame(root)
#f1.pack()
#f1.place(height=300, width=350, x =25, y= 150)

class GUImain:
    dataset = pd.DataFrame({})
    def __init__(self,master):
        self.master = master
        master.title ("Prediksi Harga Saham JST")

        #Data Path
        self.pathLabel = Label(master, text="Dataset Path ")
        self.pathEntry = Entry(master, validate ="key")
        self.browse_button = Button(master, text= "Browse", command= lambda:self.browse())
        #Datatest Path
        self.pathtestLabel = Label(master, text="Data Test Path")
        self.pathtestEntry = Entry(master, validate="key")
        self.browse_testbtn = Button(master, text= "Browse", command= lambda:self.browsetest())

        #Parameter
        #Number of Hidden Layers
        self.numhlLabel = Label(master, text="Neuron in Hidden Layers  ")
        vcmd = (root.register(self.validate),'%P')
        self.numhlEntry = Entry(master, validate="key", validatecommand=vcmd)
        # Activation Function
        self.actLabel = Label(master, text="Activation Function ")
        self.actfun = ttk.Combobox(master,values=("logistic","tanh","relu"),state='readonly')
        # Solver
        self.solvLabel = Label(master, text = 'Solver ')
        self.solv = ttk.Combobox(master,values=("lbfgs","sgd","adam"),state='readonly')
        # Number Of Iteration
        self.iterLabel = Label(master, text='Max Number of Iterations')
        self.iter = Spinbox(master, from_=0, to=10000000 )
        
        #ANN Button
        self.annbutton = Button(master, text= 'Apply',command= lambda:self.ANN())

        # Layout
        self.pathLabel.pack()
        self.pathLabel.place(x=25, y=10)   
        self.pathEntry.pack()
        self.pathEntry.place(x=170, y=10, width=130)
        self.browse_button.pack()
        self.browse_button.place(x=310, y=10,width= 60)
        
        self.pathtestLabel.pack() 
        self.pathtestLabel.place(x=25, y=40) 
        self.pathtestEntry.pack()
        self.pathtestEntry.place(x=170, y=40, width=130)
        self.browse_testbtn.pack()
        self.browse_testbtn.place(x=310, y=40,width= 60)

        self.numhlLabel.pack()
        self.numhlLabel.place(x=25, y=70)
        self.numhlEntry.pack()
        self.numhlEntry.place(x=170, y=70, width=130)    

        self.actLabel.pack()
        self.actLabel.place(x=25, y=100)     
        self.actfun.pack()
        self.actfun.place(x=170, y=100, width=130)

        self.solvLabel.pack()
        self.solvLabel.place(x=25, y=130)
        self.solv.pack()
        self.solv.place(x=170, y=130, width=130)

        self.iterLabel.pack()
        self.iterLabel.place(x=25, y=160)
        self.iter.pack()
        self.iter.place(x=170, y=160, width=130)
        
        self.annbutton.pack()
        self.annbutton.place(x= 25, y= 197, width=340)
    
    def validate(self, new_text):
        if not new_text:  # the field is being cleared
            self.entered_number = 0
            return True
        try:
            self.entered_number = int(new_text)
            return True
        except ValueError:
            return False

  
    def browse(self):
        # Clear Field
        self.pathEntry.delete(0, END) 
        # Get Data
        self.datapath = filedialog.askopenfilename(defaultextension='.csv',filetypes=[("CSV files","*.csv")])
        self.pathEntry.insert(0, self.datapath)
        
        # Validation Check
        if(not self.datapath):
            messagebox.showinfo("Neural Network Forecasting","Please Choose Data File")
            return
        # Import Data
        self.dataset = pd.read_csv(self.datapath, sep=",",index_col=0)
        windata= tk.Toplevel(root)
        windata.geometry("780x500")
        windata.title = ("Dataset")
        f2 = LabelFrame (windata)
        f2.pack(fill=BOTH, expand=True)
        f2.place(x=25,y=10,height = 480, width=720)
        self.table = self.pt = Table(f2,width=600,height=600,showtoolbar=False, showstatusbar=False )
        self.pt.importCSV(filename=self.datapath)
        self.pt.autoResizeColumns()
        self.pt.show()

    
    def browsetest(self):
        # Clear Path
        self.pathtestEntry.delete(0, END)
        #Get Data Test
        self.datatestpath = filedialog.askopenfilename(defaultextension='.csv',filetypes=[("CSV files","*.csv")])
        self.pathtestEntry.insert(0, self.datatestpath)
        #Validation
        if(not self.datatestpath):
            messagebox.showinfo("Neural Network Forecasting","Please Choose Data Test File")
            return
        self.datatest = pd.read_csv(self.datatestpath, sep=",",index_col=0)
        windatatest= tk.Toplevel(root)
        windatatest.geometry("780x500")
        windatatest.title = ("Data Test")
        f3 = LabelFrame (windatatest)
        f3.pack(fill=BOTH, expand=True)
        f3.place(x=25,y=10,height = 480, width=720)
        self.table = self.pt = Table(f3,width=600,height=600,showtoolbar=False, showstatusbar=False )
        self.pt.importCSV(filename=self.datatestpath)
        self.pt.autoResizeColumns()
        self.pt.show()
              
    def ANN(self):         
        # Drop Null Value
        self.dataset = self.dataset.dropna()
        self.datatest = self.datatest.dropna()
        # Column selection
        x_train = self.dataset.drop("Close",axis=1)
        y_train = self.dataset["Close"]
        x_test = self.datatest.drop("Close",axis=1)
        self.y_test = self.datatest["Close"]
        # Preprocess Data
        mms = preprocessing.MinMaxScaler(feature_range=(0,1))
        x_train = mms.fit_transform(x_train)
        x_test = mms.transform(x_test)
        # MLP
        hlnum=int(self.numhlEntry.get())
        maxiter = int(self.iter.get())
        actval = self.actfun.get()
        solvval = self.solv.get()
        mlp = MLPRegressor(hidden_layer_sizes=(hlnum),activation=actval,solver=solvval,max_iter=maxiter,max_fun=maxiter,random_state=0)
        mlp.fit(x_train,y_train)
        self.pred = mlp.predict(x_test)
        #GUI Hasil
        newwin = tk.Toplevel(root)
        newwin.geometry("300x600+100+50")
        pred_score = mlp.score(x_test,self.y_test)
        rmse_test = mean_squared_error(self.y_test,self.pred,squared=False)
        mse_test = mean_squared_error(self.y_test,self.pred,squared=True)
        mape_test = mean_absolute_percentage_error (self.y_test,self.pred)
        self.hsl = {'Prediction':self.pred,'Close':self.y_test,}
        self.hasil = pd.DataFrame(data=self.hsl)
        self.scoring = pd.DataFrame({'Accuracy':pred_score,'RMSE':rmse_test,'MAPE':mape_test},index=[0])
        #Table
        newwin.title = ("Hasil")
        f1 = LabelFrame(newwin)
        f1.pack()
        f1.place(x=15,y=5,height=90,width=270)
        fr = LabelFrame(newwin)
        fr.pack()
        fr.place(x=15, y=100, height=475, width=270)
        pred_score_label = Label(f1,text="Accuracy")
        mse_label = Label(f1,text="MSE")
        rmse_label = Label(f1,text="RMSE")
        mape_label = Label(f1,text="MAPE")

        pred_score_val = Label(f1,text=': %.7f'%pred_score)
        pred_score_val.pack
        pred_score_val.place(x=60, y=0)
        pred_score_label.pack()
        pred_score_label.place(x=5, y=0)
        
        mse_label.pack()
        mse_label.place(x=5, y=20)
        mse_val = Label (f1,text=': %.7f'%mse_test)
        mse_val.pack()
        mse_val.place(x=60, y=20)

        rmse_label.pack()
        rmse_label.place(x=5, y=40)
        rmse_val = Label(f1,text=': %.7f'%rmse_test)
        rmse_val.pack()
        rmse_val.place(x=60, y=40)

        mape_label.pack()
        mape_label.place(x=5, y=60)
        mape_val=Label(f1,text=': %.7f'%mape_test)
        mape_val.pack()
        mape_val.place(x=60, y=60)

        self.table = self.pt = Table(fr, dataframe=self.hasil, showtoolbar=False, showstatusbar=False, width=200)
        self.pt.autoResizeColumns()
        self.pt.show()

        self.savebtn = Button (f1, text="Save",command=lambda:self.save())
        self.savebtn.pack()
        self.savebtn.place(x=200,y=25,width=60)

        self.graphbtn = Button(f1, text="Graph",command=lambda:self.graph())
        self.graphbtn.pack()
        self.graphbtn.place(x=200, y=55, width=60)
        messagebox.showinfo("Prediksi JST","Process Complete !!!")
        return
    
    #Save
    def save(self):
        savefile =filedialog.asksaveasfilename(defaultextension='.xlsx',filetypes=[("Excel files","*.xlsx")])
        writer = pd.ExcelWriter(savefile,engine='openpyxl')
        self.scoring.to_excel(writer,sheet_name="Hasil Prediksi JST",index=False,startrow=0, startcol=0)
        self.hasil.to_excel(writer,index= True, sheet_name="Hasil Prediksi JST",startrow=3,startcol=0)
        writer.save()
        messagebox.showinfo("Save Result","Complete")
        


    #Graph
    def graph(self):
        plt.plot(self.pred,color='red', label= 'Prediction')
        plt.plot(self.y_test,color='blue', label='Close')
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.230,top=0.920)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Grafik Hasil Prediksi JST")
        plt.legend()
        plt.show()
    
my_gui =GUImain(root)
def on_closing():
    if messagebox.askokcancel("Quit","Are You Sure?"):
        root.destroy()
        os._exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()


