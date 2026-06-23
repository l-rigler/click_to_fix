import pathlib 
from gui import my_io as io
from gui import inference_pipeline as IFP 
from gui import interface_base_class as interface
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
    
class Interface(interface.interface_display):

    def __init__(self,input_folder,image_name,model_folder,label_path,fold=(0,1,2,3,4)):

        #model variables 
        self.input_folder=input_folder
        self.image_name=image_name
        self.image=io.load(input_folder+'/'+image_name)
        self.io_dict = self.image.metadata
        self.added_click=[]
        self.label_dict=io.load_labels(label_path)
        self.fold = fold
        self.labels={self.label_dict.descriptions[k]: k for k in range(len(self.label_dict.descriptions))}
        self.nlabel=len(self.label_dict.descriptions)-1 if 'ignore' in self.label_dict.descriptions else len(self.label_dict.descriptions)
        self.image_id=re.findall(r'\d+',self.image_name)[0]
        self.current_seg = None
        self.model_folder = model_folder

        #display variables
        self.slice=0
        self.current_label = 0
        self.transparency = 0.7 
        self.contrast = None
        
        self.inf_list = []
        
    def load_image_in_ram(self,):
        """function for loading image to ram
        image shape should be (x,y,z) : the sitk formalism"""
        self.ram_im = np.expand_dims(self.image.array.transpose([2,1,0]),axis=0)
        self.metadata = {'spacing' : (self.image.spacing[2],self.image.spacing[0],self.image.spacing[1])}
        # self.im_prop = {'spacing' : A.spacing} 
        for k in range(1,self.nlabel+1):
            temp_im = np.expand_dims(self.image.array.transpose([2,1,0]),axis = 0)
            temp_im = temp_im*0
            self.ram_im = np.concatenate((self.ram_im,temp_im),axis = 0)
    
    def preprocess_image(self,image):
        """nnUNet-like preprocessing of an 4D image with first dimension = channels"""
        #Z-normalization
        image[0] = (image[0]-image[0].mean())/image[0].std()

        return image
    
    def _set_predictor(self,checkpoint_name = 'checkpoint_final' ):
        self.predictor = IFP.inferer(self.model_folder,self.fold,checkpoint_name)
        self.predictor.model_init()

    def make_first_prediction(self,):
        self.ram_im = self.preprocess_image(self.ram_im)

        self.torch_input = torch.tensor(self.ram_im,device = torch.device('cuda:0'),dtype = torch.float) # setting image in gpu 
        # self.predictor.first_prediction(self.torch_input,self.metadata)
        self.predictor.first_prediction(self.torch_input)
        self.current_seg = self.predictor.current_seg
        self.inf_list.append(self.current_seg)

    def make_next_prediction(self,x,y,z):
        self.predictor.refinement_prediction_logits_manyfolds(self.torch_input,x,y,z)
        self.current_seg = self.predictor.current_seg
        self.inf_list.append(self.current_seg)
        
    def add_click(self,label,x,y,factor=1000,sigma=(2,2,2)):
        """  deprecated as we allow you to use sribble now 
            see _save_scribble as the new function """
        
        if not label.isdigit() :
            label=self.labels[label]+1
        else : label=int(label)+1

        channel = np.zeros(self.ram_im[0].shape) #temp matrix to process clicks on a specific label
        clics=[k for k in self.added_click if int(k[3]) == label-1]
        for k in clics: channel[k[2],k[1],k[0]] = 1
        channel=gaussian_filter(channel,sigma)*factor
        # self.torch_input[label] = channel
        self.torch_input[label] = torch.as_tensor(channel, dtype=torch.float, device=torch.device('cuda:0'))
        print(f'click at {x,y} added in channel number {label}')
        print('predicting...')
        # self.make_next_prediction(x,y,self.slice)
        # self.update_display(self.ax)

    def save_tensor(self,):
        out = pathlib.Path('test_inferer_debug')
        out.mkdir(exist_ok=True)
        to_save = self.torch_input.cpu().numpy().transpose(0,3,2,1)
        for k in range(self.torch_input.shape[0]):
            temp = io.Image(to_save[k],**self.io_dict)
            io.save(out/f'chan_{k}.nii.gz',temp)

    def on_click(self,event):

        if event.button == 3 :
            if event.inaxes == self.ax:
                self.x, self.y = int(event.xdata),int(event.ydata)
                print(self.x,self.y,self.current_label)
                self.added_click.append((self.x,self.y,self.slice,self.current_label))
                # self.add_click(str(self.current_label),self.x,self.y)
                
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
        channel = np.zeros(self.ram_im[0].shape)

        self.added_click += temp_scribble_map # adding all points from scribble to the point saver
        clics = [k for k in self.added_click if int(k[3])==label-1]

        for k in clics: channel[k[2],k[1],k[0]]=1
        channel = gaussian_filter(channel,sigma)*factor
        self.torch_input[label] = torch.as_tensor(channel, dtype=torch.float, device=torch.device('cuda:0'))
        

    def on_release_button(self,event):
        if event.button == 3:
            self.draw = False 
            if (hasattr(self,'scribbles_map')) : #self.scribbles_map!=[] or 
                self._save_scribble(str(self.current_label))
            print(f'click at {self.x,self.y} for the sliding window reference')
            print('predicting...')
            self.make_next_prediction(self.x,self.y,self.slice)
            self.update_display(self.ax)

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
                self.save_tensor()
                print('channels saved !')
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

    def run(self,):
        """ main function to run the interface, can prepare the image (create click chans) and allow you to add click and run multiple predictions"""
        plt.switch_backend('TkAgg')
        self.set_colormap()
        self.draw = False # initiating scribble's variable
        self.current_image = self.image.array 
        self.load_image_in_ram()
        self.make_first_prediction()
        self.fig,self.ax=plt.subplots(1,figsize=(12,12))
        self.ax.imshow(self.current_image.transpose((1,0,2))[:,:,self.slice],interpolation='nearest',cmap='gray')
        self.ax.imshow(self.current_seg.transpose((1,0,2))[:,:,self.slice], alpha=self.transparency, interpolation='nearest', cmap=self.cmap , norm = self.norm)
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

    