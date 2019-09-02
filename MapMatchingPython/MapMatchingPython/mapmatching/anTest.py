# coding=utf-8 

# import wx
# from wx.html2 import WebView


# class MyHtmlFrame(wx.Frame):
#     def __init__(self, parent, title):
#         wx.Frame.__init__(self, parent, -1, title, size=(1024, 768))
#         web_view =WebView.New(self)
#         web_view.LoadURL("D:/studying/JAVA/web/day1/table/table1.html")


# app = wx.App()
# frm = MyHtmlFrame(None, "地图匹配可视化平台")
# frm.Show()
# app.MainLoop()
# import wx
# import os

# class MyForm(wx.Frame):

#     #-------------------------------------------------------------------
#     #set the window layout
#     def __init__(self):
#         wx.Frame.__init__(self, None, wx.ID_ANY,\
#                           "Multi-file type wx.FileDialog Tutorial",\
#                           pos =(0,0), size =(410,335))
#         #def the global variance
#         global TxtCfn,Contents
#         #layout the Frame
#         panel = wx.Panel(self, wx.ID_ANY)
#         TxtCfn=wx.TextCtrl(panel,pos=(15,5),size=(200,25))
#         btnO = wx.Button(panel, label="Open",pos=(225,5),size=(70,25))
#         btnS = wx.Button(panel, label="Save",pos=(300,5),size=(70,25))
#         Contents=wx.TextCtrl(panel,pos=(15,35),size=(360,260),
#                      style=wx.TE_MULTILINE|wx.HSCROLL)
#         #bind the button event
#         btnO.Bind(wx.EVT_BUTTON, self.onOpenFile)
#         btnS.Bind(wx.EVT_BUTTON, self.onSaveFile)
#     def onOpenFile(self, event):
#         """
#         Create and show the Open FileDialog
#         """
#         dlg = wx.FileDialog(
#             self, message="Choose a file",
#             defaultFile="",
#             wildcard=wildcard1,
#             style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
#             )
#         if dlg.ShowModal() == wx.ID_OK:
#             tmp=""
#             #paths = dlg.GetPaths()
#             paths = dlg.GetPaths()
#             #print "You chose the following file(s):"
#             for path in paths:
#                 tmp=tmp+path
#             #set the value of TextCtrl[filename]
#             TxtCfn.SetValue(tmp)
#             #set the value to the TextCtrl[contents]
#             file=open(TxtCfn.GetValue())
#             Contents.SetValue(file.read())
#             file.close()
#         dlg.Destroy()
#       #def onSaveFile function
#     def onSaveFile(self,event):
#         """
#         Create and show the Save FileDialog
#         """
#         dlg = wx.FileDialog(self,
#                             message="select the Save file style",
#                             defaultFile="",
#                             wildcard=wildcard2,
#                             style=wx.SAVE
#                             )
#         if dlg.ShowModal() == wx.ID_OK:
#             filename=""
#             paths = dlg.GetPaths()
#             #split the paths
#             for path in paths:
#                 filename=filename+path
#             #write the contents of the TextCtrl[Contents] into the file
#             file=open(filename,'w')
#             file.write(Contents.GetValue())
#             file.close()
#             #show the save file path
#             TxtCfn.SetValue(filename)
#         dlg.Destroy() 
#     if __name__ == "__main__":
#         app = wx.App(False)
#         frame = MyForm()
#         frame.Show()
#         app.MainLoop()


a = input("input:")

print a



