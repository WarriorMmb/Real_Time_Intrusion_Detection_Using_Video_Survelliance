# -*- coding: utf-8 -*-
"""
Created on Tue May 28 01:14:01 2019

@author: LENOVO
"""
import pandas
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import tensorflow as tf
from sort import *
from collections import defaultdict
from pygame import mixer
from tkinter import * 
from tkinter.ttk import Combobox
import tkinter.messagebox
from datetime import datetime
from bokeh.plotting import figure,show,output_file
from bokeh.models import HoverTool,ColumnDataSource

root=Tk()
root.title('Intrusion Detection')
root.config(background='light blue')

label0 = Label(root, text="INTRUSION DETECTION SYSTEM",bg='light blue',font=('Times 23 bold'))
label0.pack(side=TOP)

label1=Label(text="Potential Intruders",bg='light blue',font=('helvetica 15 bold'))
label1.place(x=10,y=100)

classes=["person","car","bird"]
combo1=Combobox(root, values=classes,height=1,width=22)
combo1.place(x=18,y=130)

label2=Label(text="Mode of operation",bg='light blue',font=('helvetica 15 bold'))
label2.place(x=210,y=100)

camera=["webcam","usb","CCTV","offline video"]
combo2=Combobox(root, values=camera,height=1,width=30)
combo2.place(x=215,y=130)

menu = Menu(root)
root.config(menu=menu)
def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","Mohini Mohan Behera, Rashmiranjan and Girish")


menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)


subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Contributors",command=Contri)


def camera():
    global ip_address
    global intrusion_type
    intrusion_type=combo1.get()
    camera_type=combo2.get()
    if(combo2.get()=="webcam"):
        ip_address=0
        but3=Button(root,padx=5,pady=5,width=42,bg='white',fg='black',relief=GROOVE,command=lambda:intrusion(ip_address,intrusion_type),text='Run Intrusion Detection',font=('helvetica 15 bold'))
        but3.place(x=5,y=300)
    elif(combo2.get()=="CCTV"):
        label3=Label(text="Enter the ip address of CCTV camera",bg='light blue',font=('helvetica 15 bold'))
        label3.place(x=10,y=210)
        entry1=Entry(root,width=60)
        entry1.place(x=10,y=250)
        def get_ip():
            ip_address=entry1.get()
            but3=Button(root,padx=5,pady=5,width=42,bg='white',fg='black',relief=GROOVE,command=lambda:intrusion(ip_address,intrusion_type),text='Run Intrusion Detection',font=('helvetica 15 bold'))
            but3.place(x=5,y=300)
        but2=Button(root,width=5,bg='white',fg='black',relief=GROOVE,command=get_ip,text='Enter',font=('helvetica 8 bold'))
        but2.place(x=400,y=250)
        
    elif(combo2.get()=="usb"):
        ip_address=1
        but3=Button(root,padx=5,pady=5,width=42,bg='white',fg='black',relief=GROOVE,command=lambda:intrusion(ip_address,intrusion_type),text='Run Intrusion Detection',font=('helvetica 15 bold'))
        but3.place(x=5,y=300)
    
    elif(combo2.get()=="offline video"):
        label4=Label(text="Enter the video path",bg='light blue',font=('helvetica 15 bold'))
        label4.place(x=10,y=210)
        entry2=Entry(root,width=60)
        entry2.place(x=10,y=250)
        def get_ip():
            ip_address=entry2.get()
            but3=Button(root,padx=5,pady=5,width=42,bg='white',fg='black',relief=GROOVE,command=lambda:intrusion(ip_address,intrusion_type),text='Run Intrusion Detection',font=('helvetica 15 bold'))
            but3.place(x=5,y=300)
        but2=Button(root,width=5,bg='white',fg='black',relief=GROOVE,command=get_ip,text='Enter',font=('helvetica 8 bold'))
        but2.place(x=400,y=250)


but1=Button(root,width=7,bg='white',fg='black',relief=GROOVE,command=camera,text='Enter',font=('helvetica 9 bold'))
but1.place(x=450,y=128)


def exitt():
   root.destroy()


but5=Button(root,padx=5,pady=5,width=42,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but5.place(x=5,y=360)
               


def intrusion(ip_address,intrusion_type):
   
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        options = {
                'model': 'cfg/yolo.cfg',
                'load': 'bin/yolo.weights',
                'threshold': 0.25,
                'gpu': 1.0,
                'saveVideo':True
                        }
        tfnet = TFNet(options)
    
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
    
    
    def alert():
       mixer.init()
       alert=mixer.Sound('beep-07.wav')
       alert.play()
       time.sleep(0.1)
       alert.play()   
       
    
    
    
    def roipoly(image):
        
        
        CANVAS_SIZE = (600,800)
        
        FINAL_LINE_COLOR = (0, 0, 255)
        WORKING_LINE_COLOR = (0, 0, 255)
        
        class PolygonDrawer(object):
            def __init__(self, window_name):
                self.window_name = window_name # Name for our window
        
                self.done = False # Flag signalling we're done
                self.current = (0, 0) # Current position, so we can draw the line-in-progress
                self.points = [] # List of points defining our polygon
        
        
            def on_mouse(self, event, x, y, buttons, user_param):
                # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        
                if self.done: # Nothing more to do
                    return
        
                if event == cv2.EVENT_MOUSEMOVE:
                    # We want to be able to draw the line-in-progress, so update current mouse position
                    self.current = (x, y)
                elif event == cv2.EVENT_LBUTTONDOWN:
                    # Left click means adding a point at current position to the list of points
                    print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
                    self.points.append((x, y))
                elif event == cv2.EVENT_RBUTTONDOWN:
                    # Right click means we're done
                    print("Completing polygon with %d points." % len(self.points))
                    self.done = True
        
        
            def run(self):
                # Let's create our working window and set a mouse callback to handle events
                
                cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
                cv2.waitKey(1)
                cv2.setMouseCallback(self.window_name, self.on_mouse)
        
                while(not self.done):
                    # This is our drawing loop, we just continuously draw new images
                    # and show them in the named window
                    canvas =image.copy()
                    if (len(self.points) > 0):
                        # Draw all the current polygon segments
                        cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                        # And  also show what the current segment would look like
                        cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR,3)
                    # Update the window
                    cv2.imshow(self.window_name, canvas)
                    # And wait 50ms before next iteration (this will pump window messages meanwhile)
                    if cv2.waitKey(50) == 27: # ESC hit
                        self.done = True
        
                # User finised entering the polygon points, so let's make the final drawing
                
                # of a filled polygon
    #            if (len(self.points) > 0):
    #                cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
    #            # And show it
    #            cv2.imshow(self.window_name, canvas)
    #            # Waiting for the user to press any key
    #            cv2.waitKey()
    #    
                cv2.destroyWindow(self.window_name)
                return self.points
        pd = PolygonDrawer("Polygon")
        pd.run()
        return pd.points
    
    
    
    def point_inside_polygon(x,y,poly):
    
        n = len(poly)
        inside =False
    
        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y
    
        return inside
    
    df=pandas.DataFrame(columns=["Start","End"])
    capture = cv2.VideoCapture(ip_address)
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret,frame=capture.read()
    bbox =roipoly(frame)
    mot_tracker = Sort() 
    track_frame={}
    track_frame = defaultdict(lambda:0,track_frame)
    count_flag=False
    status=0
    status_list=[None,None]
    times=[]   
    while True:
        status=0
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            cv2.polylines(frame, np.array([bbox]),True, (0,255,0),3)
            results = tfnet.return_predict(frame)
            l=[]
            flag=False
    
            for color, result in zip(colors, results):
                a=[result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y'],result['confidence']]
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                if label== intrusion_type:              
                    l.append(a) 
                    a=int((int(result['topleft']['x'])+int(result['bottomright']['x']))/2)
                    b=int((int(result['topleft']['y'])+int(result['bottomright']['y']))/1.8)
                    if (point_inside_polygon(a,b,bbox)):
                        flag=True
                        print("intrusion detected")
                        alert()
                        cv2.putText(frame, "intrusion", (a,b), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                        cv2.circle(frame,(a,b), 5, (0,255,255), -1)
                        cv2.polylines(frame, np.array([bbox]),True, (0,0,255),3)
                        status=1
                        
                    else:
                        print("normal") 
                        
                    
               
            if flag==True:
                track_bbs_ids = mot_tracker.update(np.array(l))
                #print( track_bbs_ids)
                for h in  track_bbs_ids:
                    a1=(int(h[0]),int(h[1]))
                    a2=int(h[2]),int(h[3])
                    track_frame[h[4]]=track_frame[h[4]]+1                          
                    text = '{}:{:.0f}'.format('person', h[4])
                    frame = cv2.rectangle(frame,a1, a2, (0,255,255), 5)
                    frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                #print(track_frame.items())
            cv2.imshow('frame', frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            status_list.append(status)
            status_list=status_list[-2:]
            if status_list[-1]==1 and status_list[-2]==0: 
                times.append(datetime.now())             
            if status_list[-1]==0 and status_list[-2]==1: 
                times.append(datetime.now())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if status==1:
                times.append(datetime.now())
            break
    #print(status_list)
    #print(times)
    for i in range(0,len(times),2):
        df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
    df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cds=ColumnDataSource(df)
    p=figure(x_axis_type='datetime',height=150,width=650)
    p.yaxis.minor_tick_line_color=None
    p.ygrid[0].ticker.desired_num_ticks=1
    p.xaxis.axis_label = 'Time(HH:MM:SS)'
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label = 'Intrusion'
    p.yaxis.axis_label_text_font_style = "bold"
    hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
    p.add_tools(hover)
    q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)
    output_file("result.html")
    show(p)
    df.to_csv("time1.csv"),
    capture.release()
    cv2.destroyAllWindows()




root.geometry("530x500+120+120")
root.mainloop()