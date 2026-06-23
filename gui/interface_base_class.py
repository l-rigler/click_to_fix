"""nnUNet version 2.6 required
This file is a lab to create prototype and test things """

import os
import re 
import matplotlib.pyplot as plt
import tkinter as tk
import pathlib 
from gui import my_io as io
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import nnunetv2.inference.predict_from_raw_data as infer
import torch
import shutil
import numpy as np 
from matplotlib.widgets import Button, RadioButtons, Slider

class interface_display():
    """class of an interactive interface that display an image and allow you to add guidance to correct segmentation """
    def __init__(self,input_folder,image_name,temp_folder,label_path,fold=(0,1,2,3,4)):

        #model variables 
        self.input_folder=input_folder
        self.image_name=image_name
        self.temp_folder=pathlib.Path(temp_folder)
        self.temp_folder.mkdir(exist_ok=True)
        self.image=io.load(input_folder+'/'+image_name)
        self.added_click=[]
        self.label_dict=io.load_labels(label_path)
        self.fold = fold
        self.labels={self.label_dict.descriptions[k]: k for k in range(len(self.label_dict.descriptions))}
        self.nlabel=len(self.label_dict.descriptions)-1 if 'ignore' in self.label_dict.descriptions else len(self.label_dict.descriptions)
        self.image_id=re.findall(r'\d+',self.image_name)[0]
        self.current_seg = None
        
        #display variables
        self.slice=0
        self.current_label = 0
        self.transparency = 0.7 
        self.contrast = None

    def set_colormap(self):
        colors_normalized=[(r/255,g/255,b/255) for r,g,b in self.label_dict.colors]
        premap=[k[0]+(k[1],) for k in zip(colors_normalized,self.label_dict.transparency)]
        self.cmap=mcolors.ListedColormap(premap)
        boundaries = np.arange(-0.5,self.cmap.N,1)
        self.norm = mcolors.BoundaryNorm(boundaries, self.cmap.N, clip=True)

    def model_init(self,model_training_output_dir,checkpoint_name='checkpoint_final'):
        self.model_training_output_dir=model_training_output_dir
        os.environ['nnUNet_raw']=model_training_output_dir+'/nnUNet_raw'
        os.environ['nnUNet_preprocessed']=model_training_output_dir+'/nnUNet_preprocessed'
        os.environ['nnUNet_results']=model_training_output_dir+'/nnUNet_results'
        print(os.environ.get('nnUNet_raw'))
        print(os.environ.get('nnUNet_preprocessed'))
        print(os.environ.get('nnUNet_results'))
        self.predictor = infer.nnUNetPredictor(tile_step_size=0.5,
                            use_gaussian=True,
                            use_mirroring=True,
                            perform_everything_on_device=True,
                            device=torch.device('cuda:0'),
                            verbose=False,
                            verbose_preprocessing=False,
                            allow_tqdm=True)
        
        self.predictor.initialize_from_trained_model_folder(
        model_training_output_dir,
        self.fold,
        checkpoint_name=checkpoint_name+'.pth'
        )

        return "model initalized"
    
    def predict(self):
        self.predictor.predict_from_files(str(self.temp_folder),str(self.temp_folder/f'results_{len(self.added_click)}_clicks'),num_processes_preprocessing = 16,num_processes_segmentation_export = 16)
        self.last_saved = str(self.temp_folder/f'results_{len(self.added_click)}_clicks')+f'/image_{self.image_id}.nii.gz'
        self.current_seg = io.load(self.last_saved).array
        return 'image predicted!'

    def prepare_image(self):
        if any(pathlib.Path(self.temp_folder).iterdir()):
            shutil.rmtree(pathlib.Path(self.temp_folder))
            pathlib.Path(self.temp_folder).mkdir(exist_ok=True)
        io.save(pathlib.Path(self.temp_folder) / f'image_{self.image_id}_0000.nii.gz',self.image)
        array=self.image
        array.array= 0 *array.array
        for k in range(1,self.nlabel+1):
            name=f'image_{self.image_id}_'+ str(k).zfill(4)+'.nii.gz'
            io.save(pathlib.Path(self.temp_folder) / name,array)
        print('image ready for click generation')

    def add_click(self,label,x,y,factor=1000,kernel=(5,5),sigma=(2,2,2)):
        if not label.isdigit() :
            label=self.labels[label]+1
        else : label=int(label)+1
        channel=io.load(self.temp_folder / f'image_{self.image_id}_{str(label).zfill(4)}.nii.gz')
        channel.array=channel.array*0
        clics=[k for k in self.added_click if int(k[3])==label-1]
        channel.array[x,y,self.slice]=1 # inverted because torch x,y axis are in y,x in numpy
        for k in clics: channel.array[k[0],k[1],k[2]]=1
        channel.array=gaussian_filter(channel.array,sigma)*factor
        io.save(self.temp_folder / f'image_{self.image_id}_{str(label).zfill(4)}.nii.gz',channel)
        print(f'click at {x,y} added in channel number {label}')

    def on_click(self,event):

        if event.button == 3 :
            if event.inaxes == self.ax:
                x, y = int(event.xdata),int(event.ydata)
                print(x,y,self.current_label)
                self.added_click.append((x,y,self.slice,self.current_label))
                if self.current_label!=0:
                    color = self.cmap.colors[self.current_label]
                    self.ax.scatter(x,y,c = color,s=1)
                else: 
                    self.ax.scatter(x,y,c = 'black',s=1)
                self.add_click(str(self.current_label),x,y)
                plt.draw()

                #scribble part 
                self.draw = True
                self.scribbles_map = []
                x, y = int(event.xdata),int(event.ydata)
                plt.draw()
            else:
                pass

    def lets_scribbles(self,event):
        if self.draw and (event.inaxes == self.ax) : 
            x = int(event.xdata)
            y = int(event.ydata)
            self.scribbles_map.append((x,y,self.slice))
            if self.current_label!=0:
                color = self.cmap.colors[self.current_label]
                self.ax.scatter(x,y,c = color,s=1)
            else: 
                self.ax.scatter(x,y,c = 'black',s=1)
            plt.draw()

    def _save_scribble(self,label,sigma = (2,2,2),factor = 1000):

        if not label.isdigit() :
            label=self.labels[label]+1
        else : label=int(label)+1

        temp_scribble_map = [k + (label-1,) for k in self.scribbles_map]
        channel=io.load(self.temp_folder / f'image_{self.image_id}_{str(label).zfill(4)}.nii.gz')
        channel.array=channel.array*0

        self.added_click += temp_scribble_map # adding all points from scribble to the point saver
        clics=[k for k in self.added_click if int(k[3])==label-1]

        for k in clics: channel.array[k[0],k[1],k[2]]=1
        channel.array = gaussian_filter(channel.array,sigma)*factor
        io.save(self.temp_folder / f'image_{self.image_id}_{str(label).zfill(4)}.nii.gz',channel)

    def on_release_button(self,event):
        if event.button == 3:
            self.draw = False 
            if (hasattr(self,'scribbles_map')) : #self.scribbles_map!=[] or 
                self._save_scribble(str(self.current_label))


    def on_key_input(self,event):
        # print(event.key)
        if event.key in ['left','right']:
                    bounds=self.image.shape[2]-1
                    if event.key == 'left':
                        if self.slice==0:
                            pass
                        else:
                            self.slice+=-1
                            self.update_slice(event.inaxes)
                    else:
                        if self.slice==bounds:
                            pass
                        else:
                            self.slice+=1
                            self.update_slice(event.inaxes)
        
        if event.key == 'enter':
            self.predict()
            self.update_display(event.inaxes)
        if event.key == ' ':
            if len(self.added_click) == 0 : 
                pass
            else: 
                click_on_slice = [k[:2] for k in self.added_click if k[2] == self.slice]
                self.update_slice(self.ax)
                # plt.scatter(click_on_slice[0,:],click_on_slice[1,:])
                for point in click_on_slice:
                    self.ax.scatter(point[0],point[1],s= 1, c = 'r')

        if event.key == 'ctrl+right':
            if self.transparency<0.900001:
                self.transparency +=0.1
                self.update_display(event.inaxes)
        if event.key == 'ctrl+left':
            if self.transparency>0.09:
                self.transparency -=0.1
                self.update_display(event.inaxes)

        if event.key == 'backspace':
            if len(self.added_click) == 0 : 
                print('aucun clic à effacer...')
            else:
                print(f'dernier_click -> {self.added_click[-1]} supprimé')
                self.added_click = self.added_click[:-1]
                print(f'clics actuels : {self.added_click}')


    def update_slice(self,ax):
        # if ax == None :
        self.x_zoom = self.ax.get_xlim()
        self.y_zoom = self.ax.get_ylim() 
        ax = self.ax
        ax.clear()
        if self.contrast == None:
            ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice], interpolation='nearest', cmap='gray')
        else: 
            ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice], interpolation='nearest', cmap='gray',vmax = self.contrast)
        try :
            ax.imshow(self.current_seg.transpose((1,0,2))[:,:,self.slice], alpha=self.transparency, interpolation='nearest', cmap=self.cmap , norm = self.norm)
        except:
            pass
        
        ax.text(-0.1, 0.05, f"Slice: {self.slice}", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=ax.transAxes, fontsize=12, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_autoscale_on(False)
        ax.set_xlim(self.x_zoom)
        ax.set_ylim(self.y_zoom)
        plt.draw()  # Met à jour l'affichage
           
    def on_scroll(self,event):

        bounds=self.image.shape[2]-1
        
        if event.button=="down":
            if self.slice==0:
                pass
            else:
                self.slice+=-1
                self.update_slice(event.inaxes)
        if event.button=="up":
            if self.slice==bounds:
                pass
            else:
                self.slice+=1
                self.update_slice(event.inaxes)
        
    def change_label(self,event):
        if event.key =='down':
            self.current_label = (self.current_label-1) % self.nlabel
        if event.key =='up':
            self.current_label = (self.current_label+1) % self.nlabel
            

    def update_display(self, ax,with_seg=True):
        # if ax == None:
        self.x_zoom = self.ax.get_xlim()
        self.y_zoom = self.ax.get_ylim()
        ax = self.ax
        ax.clear()  # Efface l'ancienne image
        if self.contrast == None:
            ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice], interpolation='nearest', cmap='gray')
        else: 
            ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice], interpolation='nearest', cmap='gray',vmin = 0,vmax = self.contrast)
        if with_seg:
            ax.imshow(self.current_seg.transpose((1,0,2))[:,:,self.slice], alpha=self.transparency, interpolation='nearest', cmap=self.cmap, norm = self.norm,)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(-0.1, 0.05, f"Slice: {self.slice}", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=ax.transAxes, fontsize=12, color='red')
        # ax.set_autoscale_on(False)
        ax.set_xlim(self.x_zoom)
        ax.set_ylim(self.y_zoom)
        plt.draw()  # Met à jour l'affichage   
     
    def change_label_via_button(self,event):
        self.current_label = self.labels[event]
        if self.current_label !=0:
            self.label_button.activecolor = self.cmap.colors[self.current_label]
        else: 
            self.label_button.activecolor='black'
        plt.draw()
        print(self.current_label)

    def run(self,prepare_image=False):
        """ main function to run the interface, can prepare the image (create click chans) and allow you to add click and run multiple predictions"""
        plt.switch_backend('TkAgg')
        # if path==None:
        #     path=pathlib.Path('.')
        # else:
        #     path=pathlib.Path(path)
        self.set_colormap()
        self.draw = False # initiating scribble's variable
        self.current_image=io.load(self.input_folder+'/'+self.image_name).array
        if prepare_image: self.prepare_image()
        self.fig,self.ax=plt.subplots(1,figsize=(12,12))
        self.ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice],interpolation='nearest',cmap='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(-0.1, 0.05, f"Slice: {self.slice}", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=self.ax.transAxes, fontsize=12, color='red')
        
        #button for going to next image 
        self.ax_next_image = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.next_image_button = Button(self.ax_next_image,'Next image')

        #button for choosing the label of clicks
        self.label_ax = self.fig.add_axes([0.81,0.6,0.2,0.3])
        self.label_button = RadioButtons(self.label_ax,labels = list(self.labels.keys()),activecolor='black')
        self.label_button.on_clicked(self.change_label_via_button)

        def f(event):
            print('toto')
        
        self.next_image_button.on_clicked(f)
        #event connected
        
        self.fig.canvas.mpl_connect('button_press_event',self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event',self.on_key_input)
        self.fig.canvas.mpl_connect('button_release_event',self.on_release_button)
        self.fig.canvas.mpl_connect('motion_notify_event',self.lets_scribbles)

        plt.show() # it's forced to be show here because otherwise canvas never appears
        print(self.added_click)

if __name__=='__main__':
    
    # model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/model/3d_model_I_heterogenized_v2/nnUNet_results/Dataset001/nnUNetTrainerinteractive__nnUNetPlans__3d_fullres"
    model_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/model/3d_model_I_normalized/nnUNet_results/Dataset001/nnUNetTrainerinteractive__nnUNetPlans__3d_fullres"
    label_path = r"./labels.txt"
    # input_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/click_project/3D/dataset/3d_testset/for_nnunet"
    # input_folder=r'N:\0_Wip\New\1_Methodological_Developments\4_Methodologie_Traitement_Image\#8_2022_Re-Segmentation\legs\3d_model_I_fullstack_P0C_alphaexp\to_predict'
    input_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/legs/testing/hard_data/for_nnunet'
    image_name='image_1073_0000.nii.gz'
    fold=(0,)
    test=interface_display(input_folder,image_name,'./temp_interface',label_path,fold)
    test.prepare_image()
    # test.model_init(r'N:\0_Wip\New\1_Methodological_Developments\4_Methodologie_Traitement_Image\#8_2022_Re-Segmentation\legs\2d_model_I_5_75\nnUNet_results\Dataset001\interactive_2d_nnUNetTrainer__nnUNetPlans__2d')
    test.model_init(model_folder,checkpoint_name='checkpoint_final')
    test.run()
