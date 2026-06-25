import tkinter as tk 
from tkinter import ttk 
from scripts import gui_launcher_utils as GU
from gui import gui_main as GM
from tkinter import simpledialog
def run_script():
    input_path = images_folder.get()
    label_file_path = entry_label_file.get()
    model = entry_model_folder.get()
    output_path = entry_output.get()
    chk = True if chk_var.get() == 'Yes' else False
    fold = entry_fold.get()
    fold = tuple(map(int, fold.split(",")))
    #cleaning all path : 
    input_path = GU.clean_path(input_path)
    label_file_path = GU.clean_path(label_file_path)
    model = GU.clean_path(model)
    output_path = GU.clean_path(output_path)

    my_display = GM.Protocole(input_path,model,output_path,label_file_path,'image',fold = fold,chk = chk)
    my_display._set_predictor(checkpoint_name = 'checkpoint_final')
    my_display._get_image_in_temp_folder()
    my_display.load_image_in_ram()
    my_display.run()


root = tk.Tk()
root.title("Lancement du script")

config_dict = GU.load_config_dict()

tk.Label(root, text="Input path").grid(row=0, column=0)
images_folder = tk.Entry(root, width=130)
images_folder.grid(row=0, column=1)

tk.Label(root,text = 'Model folder').grid(row = 3,column = 0)
entry_model_folder = tk.Entry(root,width = 130)
entry_model_folder.grid(row = 3, column = 1 )

tk.Label(root,text = 'Label file').grid(row = 4, column = 0)
entry_label_file = tk.Entry(root,width = 130)
entry_label_file.grid(row = 4,column = 1 )

tk.Label(root, text="Output path").grid(row=1, column=0)
entry_output = tk.Entry(root, width=130)
entry_output.grid(row=1, column=1)

tk.Label(root, text="Folds").grid(row=2, column=0)
entry_fold = tk.Entry(root)
entry_fold.insert(0,'0,1,2,3,4')
entry_fold.grid(row=2, column=1)

tk.Label(root,text = 'Image common name').grid(row = 6, column = 0 )
entry_im_tag = tk.Entry(root)
entry_im_tag.insert(0,'image')
entry_im_tag.grid(row = 6, column = 1)

chk_var = tk.StringVar(value = 'Yes')
tk.Label(root,text = 'Do you want to start from where you stopped ?').grid(row = 5 ,column = 0)
chk_menu = ttk.Combobox(root,textvariable = chk_var, values = ['Yes','No'],state = 'readonly')
chk_menu.grid(row = 5,column = 1)
tk.Button(root, text="Lancer", command=run_script).grid(row=10, column=3)

config_var = tk.StringVar(value = 'None')
tk.Label(root,text = 'Configuration').grid(row = 7 ,column = 0)
chk_menu = ttk.Combobox(root,textvariable = config_var, values = ['None',] + list(config_dict.keys()),state = 'readonly')
chk_menu.grid(row = 7,column = 1)

def load_config():
    GU.load_configuration(config_var.get(),entry_model_folder,entry_label_file,entry_output,config_dict)
tk.Button(root, text = 'Load configuration',command = load_config).grid(row = 7,column = 3)

def save_config(config_dict):
    name = simpledialog.askstring('','config name to save :')
    if config_dict is None:
        print('can\'t save wihtout a config.json, please consider creating it in scripts folder ' )
    else: 
        GU.save_config_in_json(config_dict,name,entry_model_folder,entry_label_file,entry_output)
        config_dict = GU.load_config_dict()

tk.Button(root, text = 'Save configuration',command = lambda : save_config(config_dict)).grid(row = 6,column = 3)


tk.Button(root, text="Lancer", command=run_script).grid(row=10, column=3)

root.mainloop()