import tkinter as tk
from tkinter import ttk
from models import *

class AboutWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        # write something bout yourself
        

class FitPredict(tk.Tk):
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
        
        self.fitpredictFrame.pack(anchor = "nw", padx = 12, pady = 12)

        

        
