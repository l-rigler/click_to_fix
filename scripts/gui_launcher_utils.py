import tkinter as tk 
import pathlib 
import json 

def load_configuration(config_input,entry_model_folder,entry_label_file,entry_output,config_dict):
    entry_model_folder.delete(0,tk.END)
    entry_label_file.delete(0,tk.END)
    entry_output.delete(0,tk.END)
    if config_input in config_dict.keys():
        config = config_dict[config_input]
        entry_model_folder.insert(0,config['entry_model_folder'])
        entry_label_file.insert(0,config['entry_label_file'])
        entry_output.insert(0,config['entry_output'])
        print(f'{config_input} config loaded!')
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

def load_config_dict(config_dict_path = None):
    if config_dict_path is None : 
        path = pathlib.Path(__file__).resolve().parent / 'launcher_config.json'
        if not path.exists():
            return None
    else:
        path = pathlib.Path(config_dict_path)
    with open(path , "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_config_in_json(config_dict,name,entry_model_folder,entry_label_file,entry_output):

    to_dump =  {'entry_model_folder' : entry_model_folder.get(),
                    'entry_label_file' : entry_label_file.get() ,
                    'entry_output' : entry_output.get()}
    config_dict[name] = to_dump
    path = pathlib.Path(__file__).resolve().parent / 'launcher_config.json'
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"config saved under the name: {name}")

if __name__ == '__main__':
    load_config_dict()
    