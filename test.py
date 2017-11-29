# TrainAndTest.py

import cv2
import numpy as np
import operator
import os

#modula la variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # variable miembro ############################################################################
    npaContour = None           # contorno
    boundingRect = None         # delimitador recto para el contorno
    intRectX = 0                # delimitador recto  para la esquina superior izquierda en x
    intRectY = 0                # delimitador recto  para la esquina superior izquierda en y
    intRectWidth = 0            # delimitador recto ancho
    intRectHeight = 0           # delimitador recto alto
    fltArea = 0.0               # area de contorno

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calcula la informacion del delimitador recto
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # aqui se revisa que el contorno sea valido, es demasiado simple
        if self.fltArea < MIN_CONTOUR_AREA: return False        #  es necesario una mejor validacion
        return True

###################################################################################################
def main():
    allContoursWithData = []                # se declaran listas vacias,
    validContoursWithData = []              # para llenar en corto

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # lee los datos del entrenamiento (clasificaciones)
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # lee las imagenes de entrenamiento
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # se ajusta el arreglo numpy para recibir los datos de las clasificaciones

    kNearest = cv2.ml.KNearest_create()                   # instacia el objeto KNN

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("manuscrita.png")          # lee la imagen de prueba

    if imgTestingNumbers is None:                           # si la imagen no se lee correctamente
        print "error: image not read from file \n\n"        # se muestra un mensaje de error
        os.system("pause")                                  # se pausa el programa para mostrar el error
        return                                              # sale el programa
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # obtiene la imagen en escala de grises
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # la difumina

                                                            # filtra la imagen de escala de grises a blanco y negro
        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # imagen de entrada
                                          255,                                  # convierte a los pixeles que pasan el umbral en blancos
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # usa un filtro gausiano, que la media, parece tener mejores resultados
                                          cv2.THRESH_BINARY_INV,                # invierte el fondo por el primer plano  (fondo negro)
                                          11,                                   # tamanio de un pixel vecino para calcular el umbral
                                          2)                                    # constante sustraida de la media o de la media ponderada

    imgThreshCopy = imgThresh.copy()        # hace una copia de la imagen umbral, es necesario ya que findContours modifica la imagen
# imagen de entrada, asegurese de usar una copia ya que la funcion modificara esta imagen el en curso de enccontrar los contornos
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,                 # regresa el contono mas externo
                                                 cv2.CHAIN_APPROX_SIMPLE)           # comprime horizontalmente, verticalmente y diagonalmente segmentos y deja unicamente sus puntos finales

    for npaContour in npaContours:                             #  para cada contorno
        contourWithData = ContourWithData()                                             # instacia un contorno con datos del objeto
        contourWithData.npaContour = npaContour                                         # asgina el contorno al contorno con datos
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # obtiene la recta de limite
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # obtiene la informacion de la recta de limite
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calcula el area
        allContoursWithData.append(contourWithData)                                     # apendiza el contorno con datos a un arreglo de contornos con datos
    # end for

    for contourWithData in allContoursWithData:                 # para todos los contornos
        if contourWithData.checkIfContourIsValid():             # checa si es valido
            validContoursWithData.append(contourWithData)       # si lo es lo apendiza a un arreglo de contornos validos
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # ordena los contornos de izquierda a derecha

    strFinalString = ""         # declara la cadena final, esta tendra el numero final de secuencias para el final del programa

    for contourWithData in validContoursWithData:            # para cada contorno
                                                # dibuja un circulo verde en el caracter
        cv2.rectangle(imgTestingNumbers,                                        # dibuja un rectangulo en la imagen prueba original
                      (contourWithData.intRectX, contourWithData.intRectY),     # en la esquina superior izquierda
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # hasta le esquina inferior derecha
                      (0, 255, 0),              # verde
                      2)                        # delgado

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # recorta el caracter de la imagen
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # re ajusta el tamanio de la iamgen

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # aplana la imagen para introducirla en un arreglo numpy

        npaROIResized = np.float32(npaROIResized)       #convierte el vector 1D de enteros a flotantes

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 2)     # llama a la funcion knn para encontrar el mas cercano

        strCurrentChar = str(chr(npaResults[0][0]))                                             # obtiene el caracter de los resultados

        strFinalString = strFinalString + strCurrentChar            # apendiza el caracter a la cadena de salida
    # end for

    print "\n" + strFinalString + "\n"                  # muestra la cadena final

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # muestra la imagen con los rectangulos de lo que detecto
    cv2.waitKey(0)                                          # espera a que el usuario presione una tecla

    cv2.destroyAllWindows()             # borrar todas la ventanas
    

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if
