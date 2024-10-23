import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from PIL import Image
import numpy as np
import cv2
from pytesseract import Output

def QRElemination(img_path):
    image = cv2.imread(img_path)

    # Converting image to GrayScale
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_Edominer-1201_page-1.jpg",gray_img)

    # Convert gray image to blur image
    blur = cv2.GaussianBlur(gray_img,(7,7),0)
    cv2.imwrite("Blur.jpg",blur)

    # Convert blur image to thresh image(Binary image)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite("thresh.png",thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,13))
    cv2.imwrite("kernel.png",kernel)

    # Applying dilation on the kernel
    dilate = cv2.dilate(thresh,kernel,iterations=1)
    cv2.imwrite("dilate.png",dilate)

    # Creating Contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h>200 and w > 200:
            roi = image[y:y+h, x:x+w]
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),-1)
            ocr_result = pytesseract.image_to_string(roi)
            print(ocr_result)

    cv2.imwrite("QR_Eleminated_box.png",image)
    return image