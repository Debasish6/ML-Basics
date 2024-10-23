import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from PIL import Image
import numpy as np
import cv2
from pytesseract import Output
import pandas as pd
from QR_Elemination_Invoice import QRElemination

def DataPage1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    custom_config = r'-l eng --oem 1 --psm 6 '
    d = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)


    # Filter out unnecessary rows and sort the blocks
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist() 


    # Iterate through each block and extract text
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''

        for ix, ln in curr.iterrows():
            # Add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            # Calculate the number of spaces to add
            added = 0
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '
            # print(ln['text'])
            prev_left += len(ln['text']) + added + 1
        text += '\n'
        # print(text) 


    # Extract relevant information from the extracted text
    try:
        irn_text = text.split("IRN")[1].strip().split()[1] + text.split("IRN")[1].strip().split()[2]
        ack_no_text = text.split("Ack No")[1].strip().split()[2]
        ack_date_text = text.split("Ack Date")[1].strip().split()[1]
        amount_text = text.split("Amount")[1].strip().split()[4]
        bill_token = text.split("(Bill to)", 1)[1].strip()
        bill_to_text = bill_token.split("GST")[0].strip()
        gst_text_buyer = text.split("GSTIN/UIN:")[1].strip().split()[0]
        hsn_sag_text = text.split("HSN/SAG")[1].strip().split()[4]
        item_details_text = text.split("Services", 1)[1].strip().split("SUB")[0].strip()
        item_text = text.split("Services", 1)[1].strip().split()[0] + " " + text.split("Services", 1)[1].strip().split()[1]
        invoice_no_text = text.split("Invoice No")[1].strip().split()[7]
    except IndexError:
        irn_text = None
        amount_text = None
        ack_no_text = None
        bill_to_text = None
        gst_text_buyer = None
        hsn_sag_text = None
        item_text = None
        ack_date_text = None
        invoice_no_text = None

    # Print the extracted information
    print("IRN : ", irn_text)
    print("Ack No : ", ack_no_text)
    print("Ack Date : ", ack_date_text)
    print("Invoice No : ", invoice_no_text)
    print("Amount : ", amount_text)
    print("HSN/SAC : ", hsn_sag_text)
    print("Item : ", item_text)
    print("Item Details : ", item_details_text)
    print("Bill To : ", bill_to_text)
    print("GSTIN/UIN(Buyer) : ", gst_text_buyer)

    # Create a dictionary to store the extracted data
    Data = {
        'IRN': irn_text,
        'Ack No ': ack_no_text,
        'Ack Date': ack_date_text,
        'Invoice No': invoice_no_text,
        'Amount': amount_text,
        'Item': item_text,
        'Item Details': item_details_text,
        'HSN/SAC': hsn_sag_text,
        'Amount ': amount_text,
        'Bill To': bill_to_text,
        'GSTIN/UIN': gst_text_buyer,
    }

    return Data


def DataPage2(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    custom_config = r'-l eng --oem 1 --psm 6 '
    d = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)


    # Filter out unnecessary rows and sort the blocks
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist() 


    # Iterate through each block and extract text
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''

        for ix, ln in curr.iterrows():
            # Add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            # Calculate the number of spaces to add
            added = 0
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '
            # print(ln['text'])
            prev_left += len(ln['text']) + added + 1
        text += '\n' 


    # Extract relevant information from the extracted text
    try:
        total_amount_text = text.split("Total")[1].strip().split()[3]
        cgst_text =text.split("CGST")[1].strip().split()[0]
        sgst_text =text.split("SGST")[1].strip().split()[0]
        seller_text = text.split("for",1)[1].strip().split("We")[0].strip()
        item_details_text = text.split("Services", 1)[1].strip().split("CGST")[0].strip()
        # bill_token = text.split("(Bill to)", 1)[1].strip()
        # bill_to_text = bill_token.split("GST")[0].strip()
        # gst_text_buyer = text.split("GSTIN/UIN:")[1].strip().split()[0]
        # hsn_sag_text = text.split("HSN/SAG")[1].strip().split()[4]
        # item_text = text.split("Services", 1)[1].strip().split()[0] + " " + text.split("Services", 1)[1].strip().split()[1]
        # invoice_no_text = text.split("Invoice No")[1].strip().split()[7]
    except IndexError:
        total_amount_text = None
        cgst_text =None
        sgst_text =None
        seller_text = None
        item_details_text = None
        # ack_no_text = None
        # bill_to_text = None
        # gst_text_buyer = None
        # hsn_sag_text = None
        # item_text = None
        # ack_date_text = None
        # invoice_no_text = None

    # # Print the extracted information
    # print("IRN : ", irn_text)
    # print("Ack No : ", ack_no_text)
    # print("Ack Date : ", ack_date_text)
    # print("Invoice No : ", invoice_no_text)
    print("Seller : ",seller_text)
    print("CGST : ",cgst_text)
    print("SGST : ",sgst_text)
    print("Total Taxable Amount : ",float(cgst_text)+float(sgst_text))
    print("Total Amount : ", total_amount_text)
    print("Item Details : ", item_details_text)
    # print("HSN/SAC : ", hsn_sag_text)
    # print("Item : ", item_text)
    
    # print("Bill To : ", bill_to_text)
    # print("GSTIN/UIN(Buyer) : ", gst_text_buyer)

    # Create a dictionary to store the extracted data
    Data = {
        'Seller': seller_text,
        'Total Amount': total_amount_text,
        'Item Details': item_details_text,
        'CGST': cgst_text,
        'SGST': sgst_text,
        'Taxable Amount': float(cgst_text)+float(sgst_text),
    }

    return Data
