import tkinter as tk
from tkinter import ttk
import pandas as pd
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
from sklearn.model_selection import train_test_split
from idlelib.tooltip import Hovertip


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
            self.parent.postLoad()
            self.destroy()

        self.frame = ttk.Frame(self, borderwidth = 2)

        self.label = ttk.Label(self.frame, text = "Целевой столбец")
        self.label.pack(side = "left", pady = 2)
        self.entry = ttk.Combobox(self.frame, values = list(self.df.columns), state = "readonly")
        self.entry.pack(side = "left")

        self.frame.pack(padx = 4, pady = 6)

        self.button = ttk.Button(self, text = "OK", command = selectCol)
        self.button.pack(anchor = "center", pady = 6)
        self.resizable(0, 0)

class CsvPathWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__()
        
        self.parent = parent
        
        def confirm(* args):
            path = self.textEntry.get()
            self.parent.csvdf = read_csv(path)
            self.parent.csvdf.name = path.split("\\")[-1]
            SelectColWindow(self.parent, self.parent.csvdf)
            self.destroy()

        self.title("Select .csv file...")
        self.geometry("300x50")
        
        self.bind("<Return>", confirm)
                
        self.frame = ttk.Frame(self)
        
        self.textEntry = ttk.Entry(self.frame)
        self.textEntry.pack(side = "left", padx = 12, pady = 12, fill = "x", expand = True)
        self.confirmButton = ttk.Button(self.frame, text = "Confirm", command = lambda: exceptionCatcher(confirm))
        self.confirmButton.pack(side = "left", fill = "x", padx = 6)
        
        self.frame.pack(expand = True, fill = "x")
        
        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.resizable(1, 0)


class DemoWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()

        self.frame = ttk.Frame(self)

        self.loadedText = tk.StringVar(value = "Load data")
        self.loadData = ttk.Button(self.frame, textvariable = self.loadedText, command = lambda: load_data_csv(self))
        self.loadData.pack(pady = 2)

        self.loadBlobs = ttk.Button(self.frame, text = "Make blobs!", command = lambda: repeat([lambda: exceptionCatcher(self.blobdata), self.postLoad], 1))
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
                                                lambda: self.draw2Ddata(self.data_X, self.data_y, self.ax),
                                                lambda: self.drawClsfResult(self.clsf, self.ax)
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

        self.entryRngSt.set(90504)
        self.entryBlobsC.set(4)
        self.entrySamplC.set(50)

        self.fig = Figure(figsize = (5, 4), dpi = 100)
        self.fig.subplots_adjust(left = 0.01, right = 0.99, top = 0.99, bottom = 0.02)
        self.ax = self.fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(self.fig, master = self)
        self.colors = ["k", "g", "m"] + [matplotlib.cm.get_cmap("viridis")(i / 100) for i in range(0, 101, 20)]

        self.canvas.get_tk_widget().pack(side = "bottom", fill = tk.BOTH, expand = True, padx = 12, pady = 12)

        self.minsize(346, 212)

    def postLoad(self):
        for column, type_ in zip(self.data_X.columns, self.data_X.dtypes):
            {"object": lambda col: self.data_X.drop(col, axis = 1, inplace = True)}.get\
                (str(type_), lambda col: None)(column)
            
        self.data_X = {
            1: lambda: 1/0,
            2: lambda: self.data_X
        }.get(self.data_X.shape[0], lambda: PCA(2).fit_transform(self.data_X))()

        redraw_canvas(
            self.canvas,
            self.ax,
            [
                lambda: self.draw2Ddata(self.data_X, self.data_y, self.ax)
            ]
        )
        self.initClsf()
        self.meshgrids = np.meshgrid(np.linspace(*self.ax.get_xbound(), 100),
                                     np.linspace(*self.ax.get_ybound(), 100))
        try:
            self.loadedText.set(self.csvdf.name)
            del self.csvdf.name
        except:
            self.loadedText.set("Load data")

    def blobdata(self):
        self.data_X, self.data_y = make_blobs(random_state=int(self.entryRngSt.get()),
                                              centers=int(self.entryBlobsC.get()),
                                              n_samples=int(self.entrySamplC.get()))

    def drawClsfResult(self, clsf, ax):
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
            self.ax.contour(X1, X2, Z[k], levels, colors=color, linestyles=linestyles)
        # line.set_data(t, y)
        # line.set_array(X)

    def draw2Ddata(self,
                   data_X: np.array,
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

    def initClsf(self) -> None:
        self.clsf = SGDClassifier(max_iter = 1000, alpha = 1e-5)

    # def updateDataPd(self, *args) -> None:
    #     SelectColWindow(self, self.csvdf)

class FitPredict(tk.Toplevel):
    def __init__(self, model: str):
        super().__init__()

        try:
            self.model = Models[model]
        except KeyError:
            self.destroy()
            raise KeyError("Model not specified")

        self.title("ML app")
        self.geometry("855x400")
        self.minsize(855, 400)

        # Controls the data, model and basic oommands
        self.globalFrame1 = ttk.Frame(self)

        self.dataLoadFrame = ttk.Frame(self.globalFrame1, borderwidth = 2)
        self.loadCsv = ttk.Button(self.dataLoadFrame, text = "Load .csv", command = lambda: load_data_csv(self))
        self.loadCsv.pack()
        self.dataLoadFrame.pack(anchor = "nw", padx = 12, pady = (12, 2))

        self.loadStatusText = tk.StringVar(self, "Awaiting data...")
        self.loadStatus = ttk.Label(self.globalFrame1, textvariable = self.loadStatusText)
        self.loadStatus.pack(anchor = "nw", padx = 14)

        self.fitpredictFrame = ttk.Frame(self.globalFrame1, borderwidth = 2)

        self.fitButton = ttk.Button(self.fitpredictFrame, text = "Train", state = "disabled")
        self.predictButton = ttk.Button(self.fitpredictFrame, text = "Predict", state = "disabled")
        self.fitButton.pack(side = "left")
        self.predictButton.pack(side = "left")

        self.fitpredictFrame.pack(anchor = "nw", padx = 12, pady = 2)
        
        self.dataSplitFrame = ttk.Frame(self.globalFrame1, borderwidth = 2)
        
        self.trainSizeFrame = ttk.Frame(self.dataSplitFrame)
        self.trainSizeLabel = ttk.Label(self.trainSizeFrame, text = "Train size:")
        self.trainSizeLabel.pack(side = "left")
        self.trainSizeVar = tk.StringVar(self, "0.2")
        self.testSizeVar = tk.StringVar(self, "Test size: 0.8")
        self.trainSizeSpinbox = ttk.Spinbox(self.trainSizeFrame, from_ = 0.01, to = 0.99, 
                                            increment = 0.01, textvariable = self.trainSizeVar,
                                            command = lambda: self.testSizeVar.set(f"Test size: {1 - float(self.trainSizeSpinbox.get()):.2f}"),
                                            width = 4)
        self.trainSizeSpinbox.pack(side = "left")
        self.trainSizeFrame.pack(anchor = "nw", padx = 0, fill = "x", expand = True)
        self.testSizeLabel = ttk.Label(self.dataSplitFrame, textvariable = self.testSizeVar)
        self.testSizeLabel.pack(anchor = "nw", padx = 0, fill = "x", expand = True, pady = 2)
        # train_test_split
        self.dataSplitFrame.pack(anchor = "nw", padx = 14)
        
        
        
        self.globalFrame1.pack(side = "left", expand = False, fill = "y", anchor = "w")

        # Data input / predict out
        self.globalFrame2 = ttk.Frame(self)

        self.globalFrame2.pack(side = "left", 
                               expand = False, 
                               fill = "both",
                               pady = (12, 0),
                               padx = 0)

        
        # Matplotlib graphs
        self.globalFrame3 = ttk.Frame(self)
        
        self.fig = Figure(figsize = (5, 4), dpi = 100)
        self.fig.subplots_adjust(left = 0.01, right = 0.99, top = 0.99, bottom = 0.02)
        # self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.globalFrame3)
        self.canvas.get_tk_widget().pack(side = "left", 
                                         fill = "both", 
                                         expand = True, padx = 0, pady = 0)

        self.globalFrame3.pack(side = "left", expand = True, fill = "both", pady = 12, padx = 12)
        # self.globalFrame4 = ttk.Frame(self)
        # self.statusText = ttk.Label(text = "status")


    def trainEvent(self):
        ...

    def predictEvent(self):
        ...

    def postLoad(self):
        # print(self.data_X, self.data_y)
        self.fitButton["state"], self.predictButton["state"] = "normal", "normal"
        self.loadStatusText.set(f"Loaded {self.csvdf.name}")

        try:
            # map(lambda i: map(lambda j: j.destroy(), i), self.inputFrames.items())
            for i in self.inputFrames:
                for j in self.inputFrames[i]:
                    j.destroy()
                i.destroy()

            self.update()
        except (NameError, AttributeError) as e:
            pass

        self.inputFrames = {}
        for column, type_ in zip(self.data_X.columns, self.data_X.dtypes):
            frame = ttk.Frame(self.globalFrame2)
            # lb = ttk.Label(frame, text = f"{column} ({self.data_X[column].min()} - {self.data_X[column].max()})", width = 25)
            # lb.pack(side = "left")
            # ent = ttk.Entry(frame, width = 8)
            # ent.pack(side = "left")
            # ent.insert(0, f"{self.data_X[column].mean():.2f}")

            self.inputFrames[frame] = self.inpLabelObject(frame, self.data_X[column], column)
            frame.pack(anchor = "w")

            ## Only works with float data
            ## Make encoding method


        self.update()
        # self.
    
    def inpLabelObject(self, frame, series, colName, width = 25) -> tuple[tk.Label, tk.Entry or tk.Spinbox]: # type: ignore
        type_ = str(series.dtype)
        labeltext = {
            "object": lambda: "",
        }.get(type_, lambda: f" ({series.min():.2f} - {series.max():.2f})")()
        label = ttk.Label(frame, text = f"{colName}" + labeltext, width = width)
        label.pack(side = "left")

        entrytype = {
            "object": lambda: ttk.Combobox(frame, values = list(np.unique(series)), state = "readonly", width = width // 3)
        }.get(type_, lambda: ttk.Entry(frame, width =  width // 3))()

        try:
            entrytype.insert(0, f"{series.mean():.2f}")
        except:
            pass

        entrytype.pack(side = "left")

        return label, entrytype


class FitPredictSVC(FitPredict):
    def __init__(self):
        super().__init__()







def dataSplit(df: pd.DataFrame, columnName: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_X = df.drop(columnName, axis = 1)
    data_y = df[[columnName]]
    return data_X, data_y

def exceptionCatcher(callable: callable, parent: tk.Tk = None) -> None:
    try:
        callable()
    except Exception as e:
        tk.messagebox.Message(parent, default = 'ok', message = e).show()

def repeat(callables: list[callable], num: int = 5) -> None:
    for _ in range(num):
        for cmd in callables:
            cmd()

def redraw_canvas(canvas, ax, mplcallables: list[callable], *args) -> None:
    ax.clear()

    for cmd in mplcallables:
        cmd()

    canvas.draw()

def load_data_csv(parent) -> None:
    CsvPathWindow(parent)