import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from PIL import Image
import numpy as np
import cv2
from pytesseract import Output
import pandas as pd
from QR_Elemination_Invoice import QRElemination
from OCRInvoice import DataPage1,DataPage2 
from pdf2image import convert_from_path


# pdf_path = r"c:/Users/edominer/Python Project/Extracting Text from Invoice/Edominer-1201.pdf"
# images = convert_from_path("Edominer-1201.pdf")
img_path = r"c:/Users/edominer/Python Project/Extracting Text from Invoice/Edominer-1201_page-1.jpg"
# Save each page as an image file
# for i, image in enumerate(images):
#     image.save(f'output_page_{i+1}.png', 'PNG')

img = QRElemination(img_path)

data = DataPage1(img)
 
data2 = DataPage2(r"c:/Users/edominer/Python Project/Extracting Text from Invoice/Edominer-1201_page-2.jpg")

for i in data2:
    data[i]=data2[i]

for key in data:
    print(key," : ",data[key])


# print("Billing to ",data['Bill To'])




