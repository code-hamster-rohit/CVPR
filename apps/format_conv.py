import re
import json

def extract_details(content):
    
    details = {
        "deliveryNoteNumber": re.search(r"\*\*Delivery Note Number:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "lpoNumber": re.search(r"\*\*LPO Number:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "driverName": re.search(r"\*\*Driver Name:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "driverContact": re.search(r"\*\*Driver Contact:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "tankNumber": re.search(r"\*\*Tank Number:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "truckNumber": re.search(r"\*\*Truck Number:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "pointOfOrigin": re.search(r"\*\*Point of Origin:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "finalDestination": re.search(r"\*\*Final Destination:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "portOfDischarge": re.search(r"\*\*Port of Discharge:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "countryOfFinalDestination": re.search(r"\*\*Country of Final Destination:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "productName": re.search(r"\*\*Name:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "quantity": re.search(r"\*\*Quantity:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "1st weight": re.search(r"\*\*1st weight:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "2nd weight": re.search(r"\*\*2nd weight:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "Net weight": re.search(r"\*\*Net weight:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "unitPrice": re.search(r"\*\*Unit Price:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "totalPrice": re.search(r"\*\*Total Price:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "batchNumber": re.search(r"\*\*Batch Number:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
        "ocrSeals": re.search(r"\*\*OCR Seals:? ?\*\*:?\s*(.*)", content, re.IGNORECASE),
    }

    # Extract the matched groups or default to "Not provided"
    extracted_details = {key: (match.group(1).strip() if match else "Not provided") for key, match in details.items()}

    # Construct the tailored dictionary
    tailored_data = {
        "deliveryNoteNumber": extracted_details["deliveryNoteNumber"],
        "lpoNumber": extracted_details["lpoNumber"],
        "driverName": extracted_details["driverName"],
        "driverContact": extracted_details["driverContact"],
        "truck_Number" : extracted_details["truckNumber"],
        "tankNumber": extracted_details["tankNumber"],
        "pointOfOrigin": extracted_details["pointOfOrigin"],
        "finalDestination": extracted_details["finalDestination"],
        "portOfDischarge": extracted_details["portOfDischarge"],
        "1st weight": extracted_details["1st weight"],
        "2nd weight": extracted_details["2nd weight"],
        "Net weight": extracted_details["Net weight"],
        "countryOfFinalDestination": extracted_details["countryOfFinalDestination"],
        "products": [
            {
                "productName": extracted_details["productName"],
                "quantity": extracted_details["quantity"],
                "unitPrice": extracted_details["unitPrice"],
                "totalPrice": extracted_details["totalPrice"],
                "batchNumber": extracted_details["batchNumber"],
                "lpoNumber": extracted_details["lpoNumber"]
            }
        ],
        "ocrSeals": extracted_details["ocrSeals"].split(",") if extracted_details["ocrSeals"] != "Not provided" else []
    }

    return tailored_data
