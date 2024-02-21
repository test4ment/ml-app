import tkinter as tk
from tkinter import ttk
# import pandas as pd
from pandas import read_csv
from models import *

class AboutWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        # write something bout yourself

class PathWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        
        def confirm():
            try:
                path = self.textEntry.get()
                read_csv(path)
                self.destroy()
            except Exception as e:
                self.textEntry.delete(0, tk.END)
                self.textEntry.insert(0, e)
            
        self.title("Select .csv file...")
        self.geometry("300x50")

        self.frame = ttk.Frame(self)
        
        self.textEntry = ttk.Entry(self.frame)
        self.textEntry.pack(side = "left", padx = 12, pady = 12, fill = "x", expand = True)
        self.confirmButton = ttk.Button(self.frame, text = "Confirm", command = confirm)
        self.confirmButton.pack(side = "left", fill = "x", padx = 6)
        
        self.frame.pack(expand = True, fill = "x")
        
        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

        
    
class DemoWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        
        def load_data():
            PathWindow()
            
                
        self.loadedText = "Load data"
        self.loadData = ttk.Button(self, text = self.loadedText, command = load_data)
        
        self.loadData.pack(anchor = "nw")
        
        # from sklearn.linear_model import SGDClassifier
        
        # clsf = SGDClassifier()
        

class FitPredict(tk.Toplevel):
    def __init__(self, model: str):
        super().__init__()
        
        def fitEvent():
            ...
            
        def predictEvent():
            ...
        
        self.model = model
        
        self.title("ML app")
        self.geometry("400x400")
        
        self.fitpredictFrame = ttk.Frame(self, borderwidth = 2)
        self.fitButton = ttk.Button(self.fitpredictFrame, text = "Fit")
        self.predictButton = ttk.Button(self.fitpredictFrame, text = "Predict")
        self.fitButton.grid(row = 0, column = 0)
        self.predictButton.grid(row = 0, column = 1)
        self.statusText = ttk.Label(text = "status")
        
        self.fitpredictFrame.pack(anchor = "nw", padx = 12, pady = 12)

        

        
