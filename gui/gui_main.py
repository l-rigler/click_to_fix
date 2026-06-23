import pathlib 
from gui import my_io as io 
from gui import interface_2
import numpy as np 
import re
import os  
import json
import matplotlib.pyplot as plt 
from matplotlib.widgets import Button, RadioButtons, Slider
import tkinter as tk 
import time 
from scipy.ndimage import gaussian_filter
import torch 

""" class that use the matplotlib interface previously designed to build a robust test of any 
interactive model you want to evaluate.
Note that it is currently designed to evaluate click"""

def write_checkpoint(path):
    s = 'this image has been done.'
    f = open(str(path)+'/Done.txt','w')
    f.write(s)

def write_time(path,start,stop):
    s = stop-start
    f = open(str(path)+'/time.txt','w')
    f.write(str(s))
    
class Protocole(interface_2.Interface):

    def __init__(self,input_folder,model_folder,temp_folder,label_path,image_tag,fold=(0,1,2,3,4),chk=False):
        self.image_path_list=[k for k in pathlib.Path(input_folder).glob(f'{image_tag}*')]
        self.image_name_list=[str(k.stem).split('.')[0] for k in self.image_path_list] #removing extension
        self.image_ids=[re.findall(r'\d+',k)[0] for k in self.image_name_list]
        self.incrementor = 0 
        image_name = self.image_path_list[self.incrementor].name 

        super().__init__(input_folder,image_name,model_folder,label_path,fold)

        self.temp_folder = pathlib.Path(temp_folder)
        self._temp_folder_parents = self.temp_folder
        image_name = self.image_name_list[self.incrementor] #  to make folders name contiguous
        self.temp_folder = self.temp_folder / image_name
        self.temp_folder.mkdir(exist_ok=True,parents=True)
        self.chk = chk # if True we try to load the checkpoint

        self.image_name = self.image_name_list[self.incrementor] # to remove the .nii.gz after the initialisation
        self.inf_list = []

    def _get_image_in_temp_folder(self,):
        """build the folder architecture, copy images and add click chans"""
        for n_image in range(len(self.image_path_list)):
            image_name = self.image_name_list[n_image]
            image_id = self.image_ids[n_image]
            temp_folder = self._temp_folder_parents /image_name
            temp_folder.mkdir(exist_ok=True)
            image = io.load(self.image_path_list[n_image])
            io.save(pathlib.Path(temp_folder) / f'image_{image_id}_0000.nii.gz',image)
            array=image
            array.array= 0 *array.array
            for k in range(1,self.nlabel+1):
                name=f'image_{image_id}_'+ str(k).zfill(4)+'.nii.gz'
                io.save(pathlib.Path(temp_folder) / name,array)
            print('image ready for click generation')

    # def load_image_in_ram(self,id):
    #     """function for loading image to ram
    #     image shape should be (x,y,z) : the sitk formalism"""
    #     image_path = self._temp_folder_parents / self.image_name
    #     self.ram_im = np.expand_dims(io.load(image_path / f'image_{id}_0000.nii.gz').array.transpose([2,1,0]),axis=0)
    #     A = io.load(image_path / f'image_{id}_0000.nii.gz')
    #     self.im_prop = {'spacing' : (A.spacing[2],A.spacing[0],A.spacing[1])}
    #     # self.im_prop = {'spacing' : A.spacing}
    #     self.io_image = A 
    #     for k in range(1,self.nlabel+1):
    #         temp_im = io.load(image_path /  f'image_{id}_{str(k).zfill(4)}.nii.gz').array.transpose([2,1,0])
    #         temp_im = np.expand_dims(temp_im,axis = 0)
    #         self.ram_im = np.concatenate((self.ram_im,temp_im),axis = 0)

    def add_click_to_ram(self,label,x,y,factor=1000,sigma=(2,2,2)):
        if not label.isdigit() :
            label=self.labels[label]+1
        else : label=int(label)+1
        channel = np.zeros(self.ram_im[0].shape) #temp matrix to process clicks on a specific label
        clics=[k for k in self.added_click if int(k[3])==label-1]
        for k in clics: channel[k[0],k[1],k[2]] = 1
        channel=gaussian_filter(channel,sigma)*factor
        self.ram_im[label] = channel
        print(f'click at {x,y} added in channel number {label}')

    def predict(self,):
        self.predict_ram()
        to_save = self.io_image
        to_save.array = self.current_seg
        (self.temp_folder/f'results_{len(self.added_click)}_clicks').mkdir(exist_ok = True)
        io.save(str(self.temp_folder/f'results_{len(self.added_click)}_clicks')+f'/image_{self.image_id}.nii.gz',to_save)

    def display_added_click_nbr(self,):
        self.click_nbr_text = self.ax.text(0.5, 1.1, f"click added: {len(self.added_click)}", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=self.ax.transAxes, fontsize=12, color='red')
        
    def update_display(self, ax, with_seg=True):
        super().update_display(ax, with_seg)
        # self.display_added_click_nbr()
        # if (len(self.added_click) >= 10):
        #     tk.messagebox.showinfo(title = 'Image terminée !',message = 'vous avez mis 10 clicks ! changez d\'image svp ')

    def update_slice(self, ax):
        super().update_slice(ax)
        # self.display_added_click_nbr()

    def on_click(self, event):
        super().on_click(event)
        # self.click_nbr_text.remove()
        # self.display_added_click_nbr()

    def save_click_dict(self):
        D ={}
        for k,clic in enumerate(self.added_click):
            D[str(k+1)] = clic
        with open(self.temp_folder / 'click_dict.json','w') as outpath:
            json.dump(D,outpath)

    def load_checkpoint(self,):
        if self.chk == False:
            return 
        for k in range(len(self.image_name_list)):
            if os.path.exists(self._temp_folder_parents / self.image_name_list[k] /'Done.txt'):
                self.incrementor+=1
            else:
                # self.image_name = self.image_path_list[k].name
                self.image_name = self.image_name_list[k]
                self.temp_folder = self._temp_folder_parents / self.image_name
                self.image_id = self.image_ids[k]
                self.image = io.load(self.input_folder+'/'+self.image_name+'.nii.gz')
                print(f'starting back at image {self.image_name}')
                return

    def save_seg(self,):
        for iter in range(len(self.inf_list)):
            path = self.temp_folder /  str(iter)
            path.mkdir(exist_ok = True)
            self.image.array = self.inf_list[iter]
            io.save(path / f'image_{self.image_id}.nii.gz',self.image)
            print(f'seg {iter} saved !')

    def change_image(self,):
        self.incrementor+=1 
        self.save_seg()
        self.inf_list = []
        if self.incrementor >= len(self.image_path_list):
            tk.messagebox.showinfo(title='Protocole terminé',message='vous avez terminé le protocole, merci ! \n Veuillez fermer l\'interface')
        else: 
            self.save_click_dict()
            self.added_click = []
            self.image_name = self.image_name_list[self.incrementor]
            self.image_id = self.image_ids[self.incrementor]
            write_checkpoint(str(self.temp_folder))
            self.timer_stop = time.time() 
            write_time(self.temp_folder,self.timer_start,self.timer_stop)
            self.temp_folder = self._temp_folder_parents / self.image_name
            self.ax.clear()
            self.image = io.load(self.input_folder+'/'+self.image_name+'.nii.gz')
            self.current_image = self.image.array
            self.current_seg = None
            self.ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice],interpolation='nearest',cmap='gray')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.text(-0.1, 0.05, f"Slice: {self.slice}", 
                                            horizontalalignment='center', verticalalignment='center',
                                            transform=self.ax.transAxes, fontsize=12, color='red')
            # self.current_label_display = self.ax.text(1.3,0.8,f'{list(self.labels.keys())[self.current_label]}:{self.current_label}',
            #         horizontalalignment='center', verticalalignment='center',
            #         transform=self.ax.transAxes, fontsize=12, color='red')
            print(f'switching to next image... -> {self.image_name}')
            self.load_image_in_ram()
            self.make_first_prediction()
            self.update_display(self.ax,with_seg=False)
            self.timer_start = time.time()

    def next_button_activation(self,event):
        self.change_image()
       
    def update_contrast(self,value):
        """function for when the contrast slider is moved """
        self.contrast = np.percentile(self.current_image.transpose((1,0,2))[:,:,self.slice],value)
        if self.current_seg is not None : 
            self.update_display(self.ax)
        else : self.update_display(self.ax,with_seg=False)
    
    def update_transparency(self,value):
        """function for the transparency slider"""
        self.transparency = value
        if self.current_seg is not None : 
            self.update_display(self.ax)
        else : self.update_display(self.ax,with_seg=False)
    


    def run(self,path=None,prepare_image=False):
        """ main function to run the interface, can prepare the image (create click chans) and allow you to add click and run multiple predictions"""
        if self.chk:
            self.load_checkpoint()
        self.load_image_in_ram()
        self.make_first_prediction()
        plt.switch_backend('TkAgg')
        # if path==None:
        #     path=pathlib.Path('.')
        # else:
        #     path=pathlib.Path(path)
        self.set_colormap()
        self.draw = False # intiating scribble's variable
        self.current_image=io.load(self.input_folder+'/'+self.image_name+'.nii.gz').array
        if prepare_image: self.prepare_image()
        self.fig,self.ax=plt.subplots(1,figsize=(12,12))
        self.ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice],interpolation='nearest',cmap='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(-0.1, 0.05, f"Slice: {self.slice}", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=self.ax.transAxes, fontsize=12, color='red')
        self.display_added_click_nbr()

        #button for going to next image 
        self.ax_next_image = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.next_image_button = Button(self.ax_next_image,'Next image')

        #button for choosing the label of clicks
        self.label_ax = self.fig.add_axes([0.75,0.6,0.2,0.3])
        self.label_button = RadioButtons(self.label_ax,labels = list(self.labels.keys()),activecolor='black',radio_props = {'s' : [64]*self.nlabel})
                
        for n,label in enumerate(self.label_button.labels):
            if label._text =='background':
                label.set_color('black')
            else:
                label.set_color(self.cmap.colors[n])  # ou toute autre couleur

        self.label_button.on_clicked(self.change_label_via_button)

        #button for modifying contrast on the image : 
        self.contrast_ax = self.fig.add_axes([0.81,0.4,0.1,0.02])
        self.contrast_button = Slider(self.contrast_ax,'contrast: ',0,100,valinit = 100,orientation='horizontal')
        self.contrast_button.on_changed(self.update_contrast)

        #button for modifying the transparency of the segmentation : 
        self.transparency_ax = self.fig.add_axes([0.81,0.35,0.1,0.02])
        self.transparency_button = Slider(self.transparency_ax,'transparency: ',0,1,valinit = self.transparency,orientation='horizontal')
        self.transparency_button.on_changed(self.update_transparency)
        
        #box for command list 
        self.tutorial_ax = self.fig.add_axes([0.02,0.25,0.23,0.6])
        self.tutorial_ax.set_xticks([])
        self.tutorial_ax.set_yticks([])
        self.tutorial_ax.text(0.05,0.55,'supprimé le dernier clic  \u2192 [backspace] \n (si segmentation pas encore générée)')
        self.tutorial_ax.text(0.05,0.45,'afficher la position des clics\n sur la slice  \u2192 [espace]')
        self.tutorial_ax.text(0.05,0.35,'générer un clic \u2192 [clic droit]')
        self.tutorial_ax.text(0.05,0.25,'faire défiler les slices \u2192 [Molette \u2191\u2193]')
        self.tutorial_ax.text(0.05,0.15,'générer la segmentation \u2192 [enter]')
        self.tutorial_ax.text(0.05,0.05,'transparence \u2192 [Ctrl] + [\u2194]')


        #event connected
        self.fig.canvas.mpl_connect('button_press_event',self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event',self.on_key_input)
        self.fig.canvas.mpl_connect('button_release_event',self.on_release_button)
        self.fig.canvas.mpl_connect('motion_notify_event',self.lets_scribbles)
        self.next_image_button.on_clicked(self.next_button_activation)

        print(f'starting with image {self.image_id}!')
        self.timer_start = time.time()
        plt.show() # it's forced to be show here because otherwise canvas never appears
        

if __name__=='__main__':

    # model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/model/3d_model_I_heterogenized_v2/nnUNet_results/Dataset001/nnUNetTrainerinteractive__nnUNetPlans__3d_fullres"
    # label_path = r"./labels.txt"
    # input_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/dataset/3d_testset/for_nnunet"

    model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/model_thighs_cubic_loss/nnUNet_results/Dataset001/Trainer_cubic_weighted_loss__nnUNetPlans__3d_fullres"
    # model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/model_Z_boosted/nnUNet_results/Dataset001/Trainer_Z_boosted__nnUNetPlans__3d_fullres"
    label_path = r"./labels_thighs.txt"
    # input_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/test_set_improved_1612"
    input_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/test_set_improved_1201"

    fold = (0,1,2,3,4)
    # temp_folder = pathlib.Path('./SJ_protocole')
    temp_folder = pathlib.Path('./teeeeest')
    temp_folder.mkdir(exist_ok=True)

    my_display = Protocole(input_folder,model_folder,temp_folder,label_path,'image',fold = fold,chk = True)
    print('instance created')
    my_display._set_predictor(checkpoint_name = 'checkpoint_final')
    print('initialised predictor')
    my_display._get_image_in_temp_folder()
    print('images put in place')
    my_display.load_image_in_ram()
    print('image put in ram')
    my_display.run()

    