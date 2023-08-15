import cv2
from time import sleep
from PIL import Image
import cv2
import mediapipe as mp
import time
import serial







def main_app(name):


        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
        thumbCoordinate = (4, 2)

        # ser = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)
        upCount = 0
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")

        pred = 0
        while True:
            ret, frame = cap.read()
            #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            multiLandMarks = results.multi_hand_landmarks

            if multiLandMarks:
                handPoints = []
                for handLms in multiLandMarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                    for idx, lm in enumerate(handLms.landmark):
                        # print(idx,lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        handPoints.append((cx, cy))

                for point in handPoints:
                    cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)

                upCount = 0
                for coordinate in fingerCoordinates:
                    if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                        upCount += 1

                if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0]:
                    upCount += 1

                cv2.putText(img, str(upCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

            # ser.write((str(upCount) + "\n").encode())
            # cv2.imshow("Finger Counter", img)


            for (x,y,w,h) in faces:


                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                pred = 0
                if confidence > 50:

                            pred += +1
                            text = name.upper()
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            # ser.write((str(text) + "\n").encode())
                            import threading
                            import DobotDllType as dType

                            api = dType.load()

                            dType.ConnectDobot(api, "COM4", 115200)
                            dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued=1)
                            dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
                            dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 10, 55, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 10, -66, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.SetEndEffectorSuctionCup(api, True, True, isQueued=1)
                            dType.SetWAITCmd(api, 800, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 10, 55, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 90, 55, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 90, -66, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)

                            dType.SetEndEffectorSuctionCup(api, True, False, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 120, 90, 55, 10, isQueued=1)
                            dType.SetWAITCmd(api, 200, isQueued=1)
                            dType.DisconnectDobot(api)




                else:   
                            pred += -1
                            text = "UnknownFace"
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)


            cv2.imshow("image", frame)


            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred > 0 : 
                    dim =(124,124)
                    img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png") 
                      
                    # make a copy the image so that the  
                    # original image does not get affected 
                    Image1copy = Image1.copy() 
                    Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                    Image2copy = Image2.copy() 
                      
                    # paste image giving dimensions 
                    Image1copy.paste(Image2copy, (195, 114)) 
                      
                    # save the image  
                    Image1copy.save("end.png") 
                    frame = cv2.imread("end.png", 1)

                    cv2.imshow("Result",frame)
                    cv2.waitKey(5000)
                break


        cap.release()
        cv2.destroyAllWindows()
        
