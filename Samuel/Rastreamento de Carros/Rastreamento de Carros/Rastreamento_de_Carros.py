import cv2
import numpy as np
import math
import datetime

#==================== Variaveis Globais ==================== 
width = 0
height = 0
contadorEntradas = 0
contadorSaidas = 0
offSetLinhaREF = 20
#============================================================

def testaEntrada(y, coordenada_Y_Entrada, coordenada_Y_Saida):
    diferencaAbsoluta = abs(y - coordenada_Y_Entrada)
    if ((diferencaAbsoluta <= 2) and (y < coordenada_Y_Saida)):
        return 1
    else:
        return 0

def testaSaida(y, coordenada_Y_Entrada, coordenada_Y_Saida):
    diferencaAbsoluta = abs(y - coordenada_Y_Saida)
    if ((diferencaAbsoluta <= 2) and (y > coordenada_Y_Entrada)):
        return 1
    else:
        return 0


faceCascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture('vehicleTraffic480.mp4')
 
if video.isOpened():
    rval , frame = video.read()
else:
    rval = False
 
while rval:
    rval, frame = video.read()
    height = np.size(frame, 0)
    width = np.size(frame, 1)

    cars = faceCascade.detectMultiScale(frame, 1.1, 2)
 
    coordenada_Y_Entrada = (height//2) + offSetLinhaREF * 2
    coordenada_Y_Saida = (height//2) + (offSetLinhaREF * 4)
    cv2.line(frame, (0, coordenada_Y_Entrada), (width, coordenada_Y_Entrada), (255, 0, 0), 2)
    cv2.line(frame, (0, coordenada_Y_Saida), (width, coordenada_Y_Saida), (0, 0, 255), 2)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        x_centro_contorno = (x + x + w) // 2
        y_centro_contorno = (y + y + h) // 2
        pontoCentral = (x_centro_contorno, y_centro_contorno)
        cv2.circle(frame, pontoCentral, 1, (0, 0, 0), 5)

        if (testaEntrada(y_centro_contorno, coordenada_Y_Entrada, coordenada_Y_Saida)):
            contadorEntradas += 1

        if (testaSaida(y_centro_contorno, coordenada_Y_Entrada, coordenada_Y_Saida)):
            contadorSaidas += 1

    cv2.putText(frame, "Entradas: {}".format(str(contadorEntradas)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(frame, "Saidas: {}".format(str(contadorSaidas)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # show result
    cv2.imshow("Result",frame)
    cv2.waitKey(1);

video.release()
cv2.destroyAllWindows()