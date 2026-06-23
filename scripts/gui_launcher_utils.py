import tkinter as tk 
import pathlib 

def load_configuration(config_input,entry_model_folder,entry_label_file,entry_output):

    if config_input == 'Thighs':

        entry_model_folder.delete(0,tk.END)
        entry_label_file.delete(0,tk.END)
        entry_output.delete(0,tk.END)
        new_entry_model_folder =  r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/model_thighs_cubic_loss/nnUNet_results/Dataset001/Trainer_cubic_weighted_loss__nnUNetPlans__3d_fullres"
        new_entry_label_file = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/model_tester/labels_thighs.txt'
        entry_model_folder.insert(0,new_entry_model_folder)
        entry_label_file.insert(0,new_entry_label_file)
        entry_output.insert(0,r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/model_tester/GUI_test')
        print('Thighs config loaded!')

    elif config_input == 'Legs':

        entry_model_folder.delete(0,tk.END)
        entry_label_file.delete(0,tk.END)
        entry_output.delete(0,tk.END)
        new_entry_model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/model/3d_model_I_heterogenized_v2/nnUNet_results/Dataset001/nnUNetTrainerinteractive__nnUNetPlans__3d_fullres"
        new_entry_label_file = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/model_tester/labels.txt'
        entry_model_folder.insert(0,new_entry_model_folder)
        entry_label_file.insert(0,new_entry_label_file)
        entry_output.insert(0,r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/model_tester/GUI_test')
        print('Legs config loaded !')

    else : 
        entry_model_folder.delete(0,tk.END)
        entry_label_file.delete(0,tk.END)
        entry_model_folder.insert(0,'')
        entry_label_file.insert(0,'')
    
    return

def path_win_to_linux(path):
    linux_root = '/mnt/rmn_files'
    if not path.drive:
        raise ValueError(" Windows path with a Drive letter (as C:) expected")
    return str(pathlib.Path(linux_root, *path.parts[1:]))

def clean_path(path):
    win = pathlib.PureWindowsPath(path)
    if win.drive or path.startswith("\\\\"):
        return path_win_to_linux(win)
    else:
        return path
