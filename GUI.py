import tkinter as tk
from zoo_model import *


class ZOO_GUI:
    def __init__(self):

        self.main_window = tk.Tk()
        self.main_window.title("Zoo")
        self.one_frame = tk.Frame()
        self.two_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()
        self.eleven_frame = tk.Frame()
        self.twelve_frame = tk.Frame()
        self.thirteen_frame = tk.Frame()
        self.fourteen_frame = tk.Frame()
        self.fifteen_frame = tk.Frame()

        self.title_label = tk.Label(self.one_frame, text='ANIMAL TYPE PREDICTOR', fg="Blue", font=("Helvetica", 18))
        self.title_label.pack()

        self.animal_label = tk.Label(self.two_frame, text='Animal:')
        self.animal_entry = tk.Entry(self.two_frame, bg="white", fg="black", width=10)

        self.animal_label.pack(side='left')
        self.animal_entry.pack(side='left')

        self.hair_label = tk.Label(self.three_frame, text='hair:')
        self.click_hair_var = tk.StringVar()
        self.click_hair_var.set("Yes")
        self.hair_label.pack(side='left')
        self.hair_inp.pack(side='left')

        self.feathers_label = tk.Label(self.four_frame, text='Feather:')
        self.click_feathers_var = tk.StringVar()
        self.click_feathers_var.set("Yes")
        self.feathers_label.pack(side='left')
        self.feathers_inp.pack(side='left')

        self.eggs_label = tk.Label(self.five_frame, text='Eggs:')
        self.eggs_entry = tk.Entry(self.five_frame, bg="white", fg="black")

        self.eggs_label.pack(side='left')
        self.eggs_entry.pack(side='left')

        self.milk_label = tk.Label(self.six_frame, text='Milk:')
        self.milk_entry = tk.Entry(self.six_frame, bg="white", fg="black")
        # self.chol_entry.insert(0,250)
        self.milk_label.pack(side='left')
        self.milk_entry.pack(side='left')

        self.airborne_label = tk.Label(self.seven_frame, text='Airborne:')
        self.airborne_fbs_var = tk.StringVar()
        self.airborne_fbs_var.set("No")
        self.airborne_inp = tk.OptionMenu(self.seven_frame, self.click_airborne_var, "No", "Yes")
        self.airborne_label.pack(side='left')
        self.airbore_inp.pack(side='left')

        self.aquatic_label = tk.Label(self.eight_frame, text='Aquatic:')
        self.click_aquatic_var = tk.StringVar()
        self.click_aquatic_var.set("Yes")
        self.aquatic_label.pack(side='left')
        self.aquatic_inp.pack(side='left')

        self.predator_label = tk.Label(self.nine_frame, text='Predator:')
        self.predator_entry = tk.Entry(self.nine_frame, bg="white", fg="black")
        # self.thalach_entry.insert(0,'150')
        self.predator_label.pack(side='left')
        self.predator_entry.pack(side='left')

        self.toothed_label = tk.Label(self.ten_frame, text='Toothed:')
        self.click_toothed_var = tk.StringVar()
        self.click_toothed_var.set("No")
        self.toothed_inp = tk.OptionMenu(self.ten_frame, self.click_toothed_var, "Yes", "No")
        self.toothed_label.pack(side='left')
        self.toothed_inp.pack(side='left')

        self.backbone_label = tk.Label(self.eleven_frame, text='Backbone:')
        self.backbone_entry = tk.Entry(self.eleven_frame, bg="white", fg="black")
        self.backbone_label.pack(side='left')
        self.backbone_entry.pack(side='left')

        self.breathes_label = tk.Label(self.twelve_frame, text='Breathes:')
        self.click_breathes_var = tk.StringVar()
        self.breathes_label.pack(side='left')
        self.breathes_inp.pack(side='left')

        self.venomous_label = tk.Label(self.thirteen_frame, text='Venomous:')
        self.click_venomous_var = tk.StringVar()
        self.venomous_label.pack(side='left')
        self.venomous_inp.pack(side='left')

        self.fins_label = tk.Label(self.fourteen_frame, text='fins:')
        self.click_fins_var = tk.StringVar()
        self.fins_label.pack(side='left')
        self.fins_inp.pack(side='left')

        self.type_predict_ta = tk.Text(self.fifteen_frame, height=10, width=25, bg='light blue')

        self.btn_predict = tk.Button(self.fifteen_frame, text='Predict Type', command=self.predict_type)
        self.btn_quit = tk.Button(self.fifteen_frame, text='Quit', command=self.main_window.destroy)

        self.type_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()
        self.eleven_frame.pack()
        self.twelve_frame.pack()
        self.thirteen_frame.pack()
        self.fourteen_frame.pack()
        self.fifteen_frame.pack()

        tk.mainloop()

    def predict_type(self):
        result_string = ""

        self.type_predict_ta.delete(0.0, tk.END)
        animal_name = self.animal_entry.get()
        animal_hair = self.click_hair_var.get()
        if hair == "Yes":
            animal_hair = 1
        else:
            animal_hair = 0

        animal_feather_classifier = self.click_feathers_var.get()
        if animal_feather_classifier == "Yes":
            animal_feather_classifier = 1
        else:
            animal_feather_classifier = 0

        animal_eggs = self.eggs_entry.get()
        animal_milk = self.milk_entry.get()
        animal_habitat = self.airborne_inp.get()
        if animal_habitat == "Yes":
            animal_habitat = 1
        else:
            animal_habitat = 0

        animal_aqua = self.click_aquatic_var.get()
        if animal_aqua == "No":
            animal_aqua = 0
        else:
            animal_aqua = 1

        animal_predator = self.predator_entry.get()
        if animal_predator == "Yes":
            animal_predator = 1
        else:
            animal_predator = 0

        animal_backbone = self.backbone_entry.get()
        animal_toothed = self.click_toothed_var.get()
        if animal_toothed == "Yes":
            animal_toothed = 1
        else:
            animal_toothed = 0

        animal_breathes = self.click_breathes_var.get()
        animal_venomous = self.click_venomous_var.get()
        if animal_breathes == "Yes":
            animal_breathes = 1
        else:
            animal_breathes = 0

        result_string += "===Animal Classifier=== \n"
        animal_info = (animal_name, animal_hair, animal_feather_classifier, animal_eggs, animal_milk,
                       animal_habitat, animal_aqua,
                       animal_predator, animal_backbone,
                       animal_toothed, animal_breathes, animal_venomous, animal_breathes)

        type_prediction = best_model.predict([animal_info])
        disp_string = ("This prediction has an accuracy of:", str(model_accuracy))

        result = type_prediction

        if type_prediction == [0]:
            result_string = (disp_string, '\n', "0 - The animal is an insect or an invertebrate")
        else:
            result_string = (disp_string, '\n' + "1 - The animal is a mammal or an amphibian")
        self.type_predict_ta.insert('1.0', result_string)


my_zoo_GUI = ZOO_GUI()
