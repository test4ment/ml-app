import tkinter as tk
from tkinter import ttk
from models import *
from windows import FitPredict, AboutWindow, DemoWindow, exceptionCatcher
from random import sample

class MainWindow(tk.Tk):
  def __init__(self):
    
    def updModel():
      self.modelSelect["values"] = [k for k, v in Models.items() if self.mlTask.get() in v["Role"]]
      
    def modelWindowOpener():
      currModel = self.modelSelect.get()
      exceptionCatcher(lambda: FitPredict(currModel))
    
    def about():
      tmp = [i for i in self.aboutButton["text"]]
      a, b = map(tmp.index, sample(tmp, 2))
      tmp[a], tmp[b] = tmp[b], tmp[a]
      self.aboutButton["text"] = "".join(tmp)

    def demo():
      DemoWindow()
    
    super().__init__()
    
    self.title("ML app")

    self.mlTaskFrame = ttk.Frame(borderwidth = 0)
    
    self.selectTaskText = ttk.Label(self.mlTaskFrame, text = "Решаемая задача:")
    self.mlTask = ttk.Combobox(self.mlTaskFrame, state = "readonly", values = list(Tasks))
    self.selectTaskText.grid(row = 0, column = 0, padx = 12, pady = 6)
    self.mlTask.grid(row = 0, column = 1)
    
    self.mlTaskFrame.pack(anchor = "nw")

    self.modelFrame = ttk.Frame(borderwidth = 2)
    self.selectModelText = ttk.Label(self.modelFrame, text = "Модель:")
    self.modelSelect = ttk.Combobox(self.modelFrame, state = "readonly", values = list(Models.keys()), postcommand = updModel)
    self.selectModelText.grid(row = 0, column = 0, padx = 12)
    self.modelSelect.grid(row = 0, column = 1)
    
    self.modelFrame.pack(anchor = "nw")
    
    self.modelWorkerButton = ttk.Button(self, text = "Начать работу", command = modelWindowOpener)
    self.modelWorkerButton.pack(anchor = "nw", padx = 12, pady = 5, ipadx = 20)

    self.aboutDemoFrame = ttk.Frame(borderwidth = 2)
    self.aboutButton = ttk.Button(self.aboutDemoFrame, text = "About", command = about)
    self.demoButton = ttk.Button(self.aboutDemoFrame, text = "Demo", command = demo)
    
    self.aboutButton.grid(row = 0, column = 0, padx = 12)
    self.demoButton.grid(row = 0, column = 1)
    
    self.aboutDemoFrame.pack(anchor = "sw", expand = True, padx = 12, pady = 12)

    self.protocol("WM_DELETE_WINDOW", self.destroy)

    self.update()
    self.minsize(self.winfo_width() + 10, self.winfo_height())
    self.geometry("275x300")


if __name__ == '__main__':
    root = MainWindow()
    root.mainloop()
