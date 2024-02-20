import tkinter as tk
from tkinter import ttk
from models import *

class MainWindow(tk.Tk):
  def __init__(self):
    
    def updModel():
      self.modelSelect["values"] = [k for k, v in Models.items() if self.mlTask.get() in v["Role"]]
    
    super().__init__()
    
    self.title("ML app")
    self.geometry("400x400")

    self.aboutButton = ttk.Button(self, text = "About") #

    self.mlTaskFrame = ttk.Frame(borderwidth = 2)
    
    self.selectTaskText = ttk.Label(self.mlTaskFrame, text = "Решаемая задача:")
    self.mlTask = ttk.Combobox(self.mlTaskFrame, state = "readonly", values = list(Tasks))
    self.selectTaskText.grid(row = 0, column = 0)
    self.mlTask.grid(row = 0, column = 1)
    
    self.mlTaskFrame.pack(anchor = "nw")

    self.modelFrame = ttk.Frame(borderwidth = 2)
    self.selectModelText = ttk.Label(self.modelFrame, text = "Модель:")
    self.modelSelect = ttk.Combobox(self.modelFrame, state = "readonly", values = list(Models.keys()), postcommand = updModel)
    self.selectModelText.grid(row = 0, column = 0)
    self.modelSelect.grid(row = 0, column = 1)
    
    self.modelFrame.pack(anchor = "nw")

    self.aboutButton.pack(anchor = "sw", expand = True, padx = 12, pady = 12)

  

if __name__ == '__main__':
    root = MainWindow()
    root.mainloop()
