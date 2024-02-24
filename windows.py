import tkinter as tk
from tkinter import ttk

import pandas as pd
# import pandas as pd
from pandas import read_csv
from models import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
from itertools import cycle
import matplotlib.cm
from sklearn.decomposition import PCA
class AboutWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        # write something bout yourself

class SelectColWindow(tk.Toplevel):
    def __init__(self, parent, df):
        super().__init__()

        self.parent = parent
        self.df = df

        def selectCol():
            self.parent.data_X, self.parent.data_y = dataSplit(self.df, self.entry.get())
            self.parent.loadedText.set(self.parent.csvdf.name)
            self.parent.methods["postLoad"]()
            self.destroy()

        self.frame = ttk.Frame(self, borderwidth = 2)

        self.label = ttk.Label(self.frame, text = "Столбец классов")
        self.label.pack(side = "left", pady = 12)
        self.entry = ttk.Combobox(self.frame, values = list(self.df.columns), state = "readonly")
        self.entry.pack(side = "left")

        self.frame.pack(padx = 4, pady = 6)

        self.button = ttk.Button(self, text = "OK", command = selectCol)
        self.button.pack(anchor = "center")

class PathWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__()
        
        self.parent = parent
        
        def confirm():
            try:
                path = self.textEntry.get()
                self.parent.csvdf = read_csv(path)
                self.parent.csvdf.name = path.split("\\")[-1]
                self.parent.event_generate("<<updateDataPd>>", when = "tail")
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
        
        def redraw_canvas(canvas, ax, callables, *args) -> None:
            ax.clear()
            
            for cmd in callables:
                cmd()
            
            canvas.draw()
            
        def load_data():
            PathWindow(self)

        def blobdata():
                self.data_X, self.data_y = make_blobs(random_state = int(self.entryRngSt.get()),
                                                      centers = int(self.entryBlobsC.get()),
                                                      n_samples = int(self.entrySamplC.get()))

        def drawClsfResult(clsf, ax):
            X1, X2 = self.meshgrids

            num = {2: 1}.get(len(clsf.classes_), len(clsf.classes_))

            Z = []

            for _ in range(num):
                Z += [np.empty(X1.shape)]
            
            linestyles = ["dashed", "solid", "dashed"]
            
            for (i, j), val in np.ndenumerate(X1):
                x1 = val
                x2 = X2[i, j]
                p = self.clsf.decision_function([[x1, x2]])
                for k in range(num):
                    try:
                        Z[k][i, j] = p[0][k]
                    except:
                        Z[k][i, j] = p[0]

            levels = [-1.0, 0.0, 1.0]

            for k, color in zip(range(num), cycle(self.colors)):
                self.ax.contour(X1, X2, Z[k], levels, colors = color, linestyles = linestyles)
            # line.set_data(t, y)
            # line.set_array(X)

            # self.canvas.draw()
        
        def draw2Ddata(data_X: np.array, 
                       data_y: np.array, 
                       ax):
            try:
                data_X, data_y = data_X.to_numpy(), data_y.to_numpy()
            except:
                pass
            for label in np.unique(data_y):
                # print(data_y == label)
                d = data_X[np.ravel(data_y == label)]
                line = ax.scatter(d[:, 0], d[:, 1])
                
            return line
        
        def repeat(callables: list[callable], num: int = 5):
            for _ in range(num):
                for cmd in callables:
                    cmd()

        def initClsf():
            self.clsf = SGDClassifier(max_iter = 1000, alpha = 1e-5)

        def updateDataPd(* args):
            # self.csvdf
            SelectColWindow(self, self.csvdf)
            initClsf()

        def postLoad():
            self.data_X = {
                1: lambda: 1/0,
                2: lambda: self.data_X
            }.get(self.data_X.shape[0], lambda: PCA(2).fit_transform(self.data_X))()

            redraw_canvas(
                self.canvas,
                self.ax,
                [
                    lambda: draw2Ddata(self.data_X, self.data_y, self.ax)
                ]
            )
            initClsf()
            self.meshgrids = np.meshgrid(np.linspace(*self.ax.get_xbound(), 100),
                                         np.linspace(*self.ax.get_ybound(), 100))

        def exceptionCatcher(callable):
            try:
                callable()
            except Exception as e:
                tk.messagebox.Message(self, default='ok', message=e).show()

        self.methods = {
            "postLoad": postLoad,
            "exceptionCatcher": exceptionCatcher
        }

        self.frame = ttk.Frame(self)

        self.bind("<<updateDataPd>>", updateDataPd)

        self.loadedText = tk.StringVar(value = "Load data")
        self.loadData = ttk.Button(self.frame, textvariable = self.loadedText, command = load_data)
        self.loadData.pack(pady = 2)

        self.loadBlobs = ttk.Button(self.frame, text = "Make blobs!", command = lambda: repeat([lambda: exceptionCatcher(blobdata), postLoad], 1))
        self.loadBlobs.pack(pady = 2)

        self.startAnim = ttk.Button(self.frame, text = "Start",
                                    command = lambda: repeat([
                                        lambda: self.clsf.partial_fit(
                                            self.data_X, 
                                            self.data_y, 
                                            classes = np.unique(self.data_y)),
                                        lambda: redraw_canvas(
                                            self.canvas,
                                            self.ax,
                                            [
                                                lambda: draw2Ddata(self.data_X, self.data_y, self.ax),
                                                lambda: drawClsfResult(self.clsf, self.ax)
                                            ]), 
                                        self.update
                                        ], 15))
        self.startAnim.pack(pady = 2)

        self.frameRngSt = ttk.Frame(self.frame)

        self.labelRngSt = ttk.Label(self.frameRngSt, text = "Random state")
        self.labelRngSt.pack(side = "left", padx = 6)
        self.entryRngSt = ttk.Spinbox(self.frameRngSt, width = 6, from_ = 0, increment = 1, to = np.inf)
        self.entryRngSt.pack(side="left")

        self.frameRngSt.pack(pady = 2)

        self.frameBlobsC = ttk.Frame(self.frame)

        self.labelBlobsC = ttk.Label(self.frameBlobsC, text = "Blobs count")
        self.labelBlobsC.pack(side = "left", padx=6)
        self.entryBlobsC = ttk.Spinbox(self.frameBlobsC, width = 6, from_ = 2, increment = 1, to = np.inf)
        self.entryBlobsC.pack(side = "left")

        self.frameBlobsC.pack(pady = 2)

        self.frameSamplC = ttk.Frame(self.frame)

        self.labelSamplC = ttk.Label(self.frameSamplC, text = "Sample count")
        self.labelSamplC.pack(side = "left", padx = 6)
        self.entrySamplC = ttk.Spinbox(self.frameSamplC, width = 6, from_ = 10, increment = 10, to = 2 ** 31 - 1)
        self.entrySamplC.pack(side = "left")

        self.frameSamplC.pack(pady = 2)

        self.frame.pack(side = "left", padx = 12, pady = 12)
        
        # self.data_X, self.data_y = make_blobs(random_state = 42, centers = 4)

        # self.clsf = SGDClassifier(max_iter=1000, alpha=1e-5)
        
        self.fig = Figure(figsize = (5, 4), dpi = 100)
        self.ax = self.fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(self.fig, master = self)  # A tk.DrawingArea.
        self.colors = ["k", "g", "m"] + [matplotlib.cm.get_cmap("viridis")(i / 100) for i in range(0, 101, 20)]

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

def dataSplit(df: pd.DataFrame, columnName: str) -> (pd.DataFrame, pd.DataFrame):
    data_X = df.drop(columnName, axis = 1)
    data_y = df[[columnName]]
    return data_X, data_y