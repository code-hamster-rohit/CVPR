from ultralytics import YOLO
import google.generativeai as genai
import os, shutil, PIL, json

class LicensePlateDetector:
    def __init__(self, model_path, temp_folders):
        self.model = YOLO(model_path)
        self.temp_input_folder = temp_folders["input"]
        self.temp_output_folder = temp_folders["output"]
        self.temp_cropped_folder = temp_folders["cropped"]
        self._initialize_folders()

    def _initialize_folders(self):
        for folder in [self.temp_input_folder, self.temp_output_folder, self.temp_cropped_folder]:
            os.makedirs(folder, exist_ok=True)

    def clear_temp_folders(self):
        for folder in [self.temp_input_folder, self.temp_output_folder, self.temp_cropped_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

    def process_image(self, image_path):
        results = self.model.predict(source=image_path, save=True, save_crop=True, project="temp_runs", name="predict")
        cropped_image_path = None
        processed_image_path = None

        try:
            list_of_predict_dirs = [d for d in os.listdir("temp_runs") if d.startswith("predict")]
            list_of_predict_dirs.sort()
            result_dir = os.path.join("temp_runs", list_of_predict_dirs[-1])

            input_filename = os.path.basename(image_path)
            processed_image_path_potential = os.path.join(result_dir, input_filename)
            if os.path.exists(processed_image_path_potential):
                 processed_image_dest = os.path.join(self.temp_output_folder, input_filename)
                 shutil.copy(processed_image_path_potential, processed_image_dest)
                 processed_image_path = processed_image_dest

            crops_dir = os.path.join(result_dir, "crops")
            if os.path.exists(crops_dir):
                for class_folder in os.listdir(crops_dir):
                    class_folder_path = os.path.join(crops_dir, class_folder)
                    if os.path.isdir(class_folder_path):
                        for crop_item in os.listdir(class_folder_path):
                            if crop_item.lower().endswith(('.png', '.jpg', '.jpeg')):
                                src_crop_path = os.path.join(class_folder_path, crop_item)
                                cropped_image_path = os.path.join(self.temp_cropped_folder, crop_item)
                                shutil.copy(src_crop_path, cropped_image_path)
                                print(f"Cropped image saved to: {cropped_image_path}")
                                break
                    if cropped_image_path:
                         break

        except (FileNotFoundError, IndexError, Exception) as e:
             print(f"Error finding prediction results/crops: {e}")
             print(f"Please check the 'temp_runs' directory structure.")

        return cropped_image_path, processed_image_path

class TextExtractor:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        except Exception as e:
            print(f"Failed to configure Google AI: {e}")
            raise

        self.model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini model initialized successfully.")

    def extract_text(self, image_path):
        print(f"Attempting to extract text from: {image_path}")
        if not os.path.exists(image_path):
             print(f"Error: Image path does not exist: {image_path}")
             return None

        try:
            with PIL.Image.open(image_path) as i:
                img = i
            i.close()
            prompt = """Extract the alphabets, numbers, and any special characters visible on the license plate in the image.
Respond ONLY with a JSON object containing a single key "license_plate_text" whose value is the extracted string.
Example JSON Response:
{
  "license_plate_text": "KA01AB1234"
}
Do not include any other text, explanations, or markdown formatting like ```json.
"""

            print("Sending request to Gemini API...")
            response = self.model.generate_content([prompt, img])

            print("Received response from Gemini.")
            try:
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned_text)
                extracted_text = data.get("license_plate_text")

                if extracted_text:
                    print(f"Successfully extracted text: {extracted_text}")
                    return extracted_text
                else:
                    print("Error: 'license_plate_text' key not found in JSON response.")
                    return cleaned_text

            except json.JSONDecodeError:
                print(f"Error: Gemini response was not valid JSON: {response.text.strip()}")
                return response.text.strip()
            except Exception as e:
                 print(f"An error occurred during JSON parsing: {e}")
                 return response.text.strip()


        except FileNotFoundError:
            print(f"Error: Cannot find image file at {image_path}")
            return None
        except Exception as e:
            print(f"An error occurred during Gemini API call or processing: {e}")
            return None

def pipeline(uploaded_image_path):
    model_path = r"model\license_plate_detector.pt"
    temp_folders = {
        "input": "temp_input",
        "output": "temp_output",
        "cropped": "temp_cropped"
    }

    for folder in temp_folders.values():
        os.makedirs(folder, exist_ok=True)

    try:
        detector = LicensePlateDetector(model_path, temp_folders)
        extractor = TextExtractor()
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        return
    
    uploaded_image = PIL.Image.open(uploaded_image_path)

    if uploaded_image:
        detector.clear_temp_folders()
        if os.path.exists("temp_runs"):
             shutil.rmtree("temp_runs")
        os.makedirs("temp_runs", exist_ok=True)

        input_image_path = os.path.join(detector.temp_input_folder, os.path.basename(uploaded_image_path))
        with open(input_image_path, "wb") as f:
            uploaded_image.save(f, format="JPEG")

        print(f"Uploading image for processing: {input_image_path}")
        cropped_image_path, processed_image_path = detector.process_image(input_image_path)

        if cropped_image_path:
            extracted_text = extractor.extract_text(cropped_image_path)

        detector.clear_temp_folders()
        if os.path.exists("temp_runs"):
            shutil.rmtree("temp_runs")
        
        if extracted_text:
            print(f"Extracted Text: {extracted_text}")
            return extracted_text
        else:
            if processed_image_path:
                print(f"Error extracting text from: {cropped_image_path}")
                return f"Error extracting text from: {cropped_image_path}"
            else:
                print(f"Error processing image: {input_image_path}")
                return f"Error processing image: {input_image_path}"