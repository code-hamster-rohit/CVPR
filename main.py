from apps.anpr_app import pipeline
from apps.dn import process_delivery_note
from apps.packmat import run
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

def anpr():
    image_paths = []
    for root, dirs, files in os.walk("inputs/anpr"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    
    license_plate_number = pipeline(image_paths[0])
    if license_plate_number:
        if not os.path.exists("inventory"):
            os.makedirs("inventory")
        if os.path.exists(f"inventory/{license_plate_number}.csv"):
            return license_plate_number
        df = pd.DataFrame(
            {
                'SR_NO': [],
                'NAME': [],
                'QUANTITY': [],
                'UNIT_PRICE': [],
                'TOTAL_PRICE': [],
                'BATCH_NUMBER': [],
                'LPO_NUMBER': [],
            }
        )
        df.to_csv(f"inventory/{license_plate_number}.csv", index=False)

    return license_plate_number

def dn(license_plate_number):
    image_paths = []
    for root, dirs, files in os.walk("inputs/bills"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    
    content = process_delivery_note(image_paths[0])

    temp_df = pd.DataFrame(content)
    csv_path = f"inventory/{license_plate_number}.csv"
    os.makedirs("inventory", exist_ok=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = temp_df.iloc[0:0].copy()

    df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    return license_plate_number


if __name__ == "__main__":
    license_plate_number = anpr()
    dn(license_plate_number)
    run()