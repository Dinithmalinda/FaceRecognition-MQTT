import numpy as np
import pickle
import os
import cv2
import time
import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

curr_path = os.path.join(os.getcwd())
print (curr_path)

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer_path = os.path.join(curr_path, 'model', 'recognizer.pickle')
le_path = os.path.join(curr_path, 'model', 'le.pickle')

recognizer = pickle.loads(open(recognizer_path, "rb").read())
le = pickle.loads(open(le_path, "rb").read()) 


def Decodecam(ip,disp,frames):   
    facecount=0
    
    returnva=""

    global le
    global face_detector
    global face_recognizer 

    
    vs=cv2.VideoCapture(ip)
    time.sleep(1)

    returnval=[]
    returnval.append("0")

    while True:
        frames-=1
        if(frames==0):break
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()
        DF=0
        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.5:
                DF+=1
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                if(disp):
                    imagelink=os.path.join(curr_path, 'tempSaveimages', 'saved'+str(i)+'.jpg')
                    cv2.imwrite(imagelink, face)
                    imagelink=os.path.join(curr_path, 'tempSaveimages', 'saved_full.jpg')
                    cv2.imwrite(imagelink, img=frame)
                

                face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                # if(proba>0.4):
                #     break
                
                if(proba> 0.6):
                    flag=True
                    for zz in returnval:
                       
                        if(zz==name):
                            flag=False

                    if flag:
                         returnval.append(name)

                if DF>facecount:
                    facecount=DF
                    returnval[0]=str(facecount)

                    
                text = "{}: {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if(disp):
            cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    return returnval





def training():
    print('DoTrainning')
    data_base_path = os.path.join(curr_path, 'ProfilePhotos')

    global face_detector
    global face_recognizer 

    filenames = []
    for path, subdirs, files in os.walk(data_base_path):
        for name in files:
            filenames.append(os.path.join(path, name))

    face_embeddings = []
    face_names = []
 
    for (i, filename) in enumerate(filenames):
        print("Processing image {}".format(filename))

        image = cv2.imread(filename)
        image = imutils.resize(image, width=600)

        (h, w) = image.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        i = np.argmax(face_detections[0, 0, :, 2])
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

            face_recognizer.setInput(face_blob)
            face_recognitions = face_recognizer.forward()

            name = filename.split(os.path.sep)[-2]

            face_embeddings.append(face_recognitions.flatten())
            face_names.append(name)


    data = {"embeddings": face_embeddings, "names": face_names}

    le = LabelEncoder()
    labels = le.fit_transform((data["names"]))

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    recognizer_path = os.path.join(curr_path, 'model', 'recognizer.pickle')
    f = open(recognizer_path , "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    le_path = os.path.join(curr_path, 'model', 'le.pickle')
    f = open(le_path, "wb")
    f.write(pickle.dumps(le))
    f.close()

#training()
#Decodecam(0,True,130)   
