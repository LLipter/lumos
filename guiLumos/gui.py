import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import *
from tkinter import *
from tkinter.messagebox import showinfo, askquestion, showerror
import os

# 设置窗口
window = tk.Tk()
window.title('Lumos视频风格变换')
window.resizable(False,False)
window.geometry("500x300")

background = ImageTk.PhotoImage(file = 'icon/back.jpg')
w = Canvas(window,width=500,height=300)
w.create_image(250,150,image=background)
w.pack()


frame = Frame(window)
frame.pack(expand=YES,fill=BOTH)
# 设置布局
# 风格
fm1 = Frame(frame)

# 视频框
fm3 = Frame(frame)

#  原视频路径
path = StringVar()

# 转换后视频路径
path_after = StringVar()
w.create_text(50,50,text='文件路径')

l4 = tk.Label(window,text='保存路径')
l4.pack(side=TOP,anchor=W)
w.create_text(50,100,text='保存路径')

entry1 = tk.Entry(window,textvariable=path,state='disabled',width=30)
entry1.pack(side=TOP,expand=YES)
w.create_window(120,70,window=entry1)

entry2 = tk.Entry(window,textvariable=path_after,state='disabled',width=30)
entry2.pack(side=TOP,expand=YES)
w.create_window(120,120,window=entry2)

style = StringVar()
pic_name = Label(window,text='风格预览:',anchor="center")
pic_name.pack(side=BOTTOM,anchor=CENTER)
w.create_text(310,280,text='风格预览')
image = ImageTk.PhotoImage(file = 'style/candy.jpg')

entry3 = tk.Entry(window,textvariable=style,state='disabled',width=10)
entry3.pack()
w.create_window(380,280,window=entry3)

pic = Label(window, bg='white',image=image,width=230, height=230,relief=SUNKEN,borderwidth=2)
pic.pack(side=BOTTOM,anchor=CENTER)
w.create_window(350,150,window=pic)

# 风格选择
def open_pic(label):
    global image
    style1 = label
    style.set(style1)
    image=ImageTk.PhotoImage(file = 'style/'+ label + '.jpg')
    pic.config(image=image)

# 选择保存路径
def save():
    path_ = askdirectory()
    path_after.set(path_)


# 关于
def about_lumos():
   top = Toplevel()
   top.title('关于Lumos')
   top.geometry('200x180')
   string = 'Lumos是一个视频转换软件，实现给视频加滤镜的功能。主要技术包括基于Tensorflow的模型训练，以及基于ffmpeg的视频处理过程。'
   l = tk.Label(top,width=200,height=180,text=str(string),wraplength=130)
   l.pack()

# 打开文件
def open_file():
    path_ = askopenfilename(filetypes=[("MP4",".mp4")])
    path.set(path_)

# 退出
def exit():
    askquestion("exit",'确定退出？')
    window.destroy()



# create menu
# 文件菜单，打开保存视频
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0) # 下拉
menubar.add_cascade(label='文件', menu=filemenu)
filemenu.add_command(label='打开', command=open_file)
filemenu.add_command(label='保存', command=save)
filemenu.add_separator()
filemenu.add_command(label='退出', command=exit)

# 编辑菜单，选择风格类型
editmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='选择风格', menu=editmenu)
# for each in ['CANDY','CUBIST','DENOISED_STARRY','FEATHERS','GOUACHE','MOSAIC','PAINTING','PICASSO','SCREAM','STARRY','UDNIE','WAVE']:
editmenu.add_radiobutton(label='candy',command=lambda:open_pic('candy'))
editmenu.add_radiobutton(label='cubist',command=lambda:open_pic('cubist'))
editmenu.add_radiobutton(label='denoised_starry',command=lambda:open_pic('denoised_starry'))
editmenu.add_radiobutton(label='feathers',command=lambda:open_pic('feathers'))
editmenu.add_radiobutton(label='gouache',command=lambda:open_pic('gouache'))
editmenu.add_radiobutton(label='mosaic',command=lambda:open_pic('mosaic'))
editmenu.add_radiobutton(label='painting',command=lambda:open_pic('painting'))
editmenu.add_radiobutton(label='picasso',command=lambda:open_pic('picasso'))
editmenu.add_radiobutton(label='scream',command=lambda:open_pic('scream'))
editmenu.add_radiobutton(label='starry',command=lambda:open_pic('starry'))
editmenu.add_radiobutton(label='udnie',command=lambda:open_pic('udnie'))
editmenu.add_radiobutton(label='wave',command=lambda:open_pic('wave'))


# 关于菜单
aboutmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='关于', menu=aboutmenu)
aboutmenu.add_command(label='关于Lumos', command=about_lumos)

# 显示视频转换信息

def preview():
    filename = str(path.get())
    if filename and os.path.exists(filename):
         try:
            os.system(filename)
         except Exception as e:
            showerror("error",e)
    else:
        showinfo("提示",'请选择打开文件')






##########
# 转换函数
def transfer():
    showinfo("transfer",'success')








def view():
    file = str(path_after.get())
    if file and os.path.exists(file):
         try:
            os.system(file)
         except Exception as e:
            showerror("error",e)
    else:
        showerror("error",'文件不存在！')


preview_file = Button(window,text='预览视频',relief=RAISED,width=15,command=preview)
preview_file.pack(side=LEFT,anchor=CENTER)
w.create_window(100,155,window=preview_file)

transfer_file = Button(window,text='开始转换',relief=RAISED,width=15,command=transfer)
transfer_file.pack(side=LEFT,anchor=CENTER)
w.create_window(100,185,window=transfer_file)

view_file = Button(window,text='查看效果',relief=RAISED,width=15,command=view)
view_file.pack(side=LEFT,anchor=CENTER)
w.create_window(100,215,window=view_file)

fm3.pack(side=LEFT,anchor=W,padx=20)
fm1.pack(side=LEFT,padx=10,ipadx=10)


window.config(menu=menubar)
window.mainloop()



