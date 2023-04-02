import opencv
from time import sleep
#import AWSMQTT
import PahoMQTT
import portscanner
import json
LAST_Datalist =[]
NEW_Datalist=[]
Devicelist={}

print("search for camera")
#CAM_PILIST=portscanner.getcamip()
#currentcam= "rtsp://admin:PNJCLT@"+CAM_PILIST+":554/H.264" # capture detected faces using opencv IPCAM Address
currentcam=0

print(currentcam)
#AWSMQTT.test()
#AWSCLIENT.PUBLISH("IPCAMdata","test123")
#print(str(opencv.Decodecam(currentcam,True,30)))
while 1:
    
    try:
        NEW_Datalist=opencv.Decodecam(currentcam,True,60)
    except:
        print("Video error")

    if NEW_Datalist!=LAST_Datalist:
        LAST_Datalist=NEW_Datalist
        xx={"count":LAST_Datalist[0],"data":LAST_Datalist}
        try:
            #AWSMQTT.PUBLISH("IPCAMdata",xx)
            PahoMQTT.PUBLISH("IPCAMdata",xx)
        except:
            print("Broker connect error")
        print(xx)
    sleep(5)
    



