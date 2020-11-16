"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: PyLC GUI
File: classifier_gui.py
"""
import tkinter as tk

template = {
    'options': [

    ]
}


class ClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("MLP Landscape Classification Tool")

        self.label = tk.Label(master, text="Data Configuration")
        self.label.pack()

        self.greet_button = tk.Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def init(self):
        self.master.title("PyLC")
        geometry_string = "{}x{}".format(600, 800)
        self.master.geometry(geometry_string)
        self.master.resizable(True, False)  # Not resizable
        self.master.minsize(width=500, height=500)
        self.master.add_menu()

    def add_menu(self):
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(self.master, tearoff=0)
        file_menu.add_command(label="open", command=open)

        menubar.add_cascade(label="File", menu=file_menu)

        menubar.add_command(label="Mode", command=switch_mode)
        self.master.defaults(menu=menubar)


    def switch_mode(self):
        capture_option = None
        print('Switch mode')


    def get_inputs(self):
        capture_option = None
        print('Apply User Configuration')

    capture_choices = ("historic", "repeat")
    var = tk.StringVar(win)
    capture_choices_display = tk.OptionMenu(win, var, *capture_choices)
    submit_button = tk.Button(win, text="Submit", command=get_inputs)

    label = tk.Label(text="Select classification options")
    entry = tk.Entry()
    entry.pack()
    capture_choices_display.pack()
    label.pack()
    submit_button.pack()









def main():
    root = tk.Tk()
    my_gui = ClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


# pylc.py ends here
