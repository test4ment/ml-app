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
    def __init__(self, parent):
        super().__init__()
        
        self.parent = parent
        
        def confirm():
            try:
                path = self.textEntry.get()
                read_csv(path)
                self.parent.path = path
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
            PathWindow(self)
            
        def anim():
            self.clsf.partial_fit(self.data_X, self.data_y, classes = np.unique(self.data_y))
            
            xx = np.linspace(*self.ax.get_xbound(), 100)
            yy = np.linspace(*self.ax.get_ybound(), 100)

            self.ax.clear()
            
            X1, X2 = np.meshgrid(xx, yy)
            Z1, Z2, Z3 = np.empty(X1.shape), np.empty(X1.shape), np.empty(X1.shape)
            
            drawblobs(self.ax)
            
            for (i, j), val in np.ndenumerate(X1):
                x1 = val
                x2 = X2[i, j]
                p = self.clsf.decision_function([[x1, x2]])
                Z1[i, j] = p[0][0]
                Z2[i, j] = p[0][1]
                Z3[i, j] = p[0][2]
            levels = [-1.0, 0.0, 1.0]
            self.ax.contour(X1, X2, Z1, levels, colors = "k", linestyles = ["dashed", "solid", "dashed"])
            self.ax.contour(X1, X2, Z2, levels, colors = "r", linestyles = ["dashed", "solid", "dashed"])
            self.ax.contour(X1, X2, Z3, levels, colors = "b", linestyles = ["dashed", "solid", "dashed"])

            # line.set_data(t, y)

            self.canvas.draw()
        
        def drawblobs(ax):
            for label in np.unique(self.data_y):
                d = self.data_X[self.data_y == label]
                self.line = ax.scatter(d[:, 0], d[:, 1])
        
        def animLoop(iter = 5):
            for _ in range(iter):
                anim()
                self.update()
            
        self.frame = ttk.Frame(self)
        
        self.loadedText = "Load data"
        self.loadData = ttk.Button(self.frame, text = self.loadedText, command = load_data)
        
        self.startAnim = ttk.Button(self.frame, text = "Start", command = lambda: animLoop(40))
        self.loadData.pack()
        self.startAnim.pack()
        
        self.frame.pack(side = "left", padx = 12, pady = 12)
        
        
        from sklearn.linear_model import SGDClassifier
        from sklearn.datasets import make_blobs
        
        self.data_X, self.data_y = make_blobs(random_state = 42)
        
        self.clsf = SGDClassifier(max_iter = 1000)
        
        from matplotlib.backend_bases import key_press_handler
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.figure import Figure
        
        import numpy as np
        
        self.fig = Figure(figsize = (5, 4), dpi = 100)
        self.ax = self.fig.add_subplot()

        drawblobs(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side = "bottom", fill = tk.BOTH, expand = True, padx = 12, pady = 12)
        

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

        

        
