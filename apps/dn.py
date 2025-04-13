from google import generativeai as genai
from pdf2image import convert_from_path
from apps.format_conv import extract_details
from fpdf import FPDF
import os, json, PIL

class DeliveryNoteTextExtractor:
    def __init__(self):
        
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            print("Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error configuring Google AI or initializing model: {e}")
            raise

    def extract_text(self, file_path):
        is_temp_image = False
        try:
            if not os.path.exists(file_path):
                 return f"Error: Image file not found at {file_path}"

            print(f"Processing image for Gemini: {file_path}")
            img = PIL.Image.open(file_path)

            prompt = """Analyze the attached delivery note image and extract the following details.
Present the information clearly. If a field is not found, indicate 'Not Found'.

Return the result in form of a code block in the following format:

```json
{
  "SR_NO": [FR01, FR02],
  "NAME": ["Item 1", "Item 2"],
  "QUANTITY": [1, 2],
  "UNIT_PRICE": [1.0, 2.0],
  "TOTAL_PRICE": [1.0, 2.0],
  "BATCH_NUMBER": ["AB001", "BC22C"],
  "LPO_NUMBER": ["AD2227AA", "BASASA11"]
}
```

NAME is the description coloumn in the image. SR_NO is the Item Code coloumn in the image.
Respond ONLY with the extracted information in the format listed above. Do not add any extra commentary before or after the list.
"""

            print("Sending request to Gemini API...")
            response = self.model.generate_content([prompt, img])

            print("Received response from Gemini.")
            if response.text:
                try:
                    cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                    data = json.loads(cleaned_text)
                    if data:
                        return data
                    else:
                        return "Error: No content returned from Gemini API. Possible safety block or API issue."
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON response: {e}")
                    return "Error: No content returned from Gemini API. Possible safety block or API issue."
            else:
                try:
                    print("Gemini response was empty. Checking feedback:", response.prompt_feedback)
                except (AttributeError, IndexError):
                     pass
                return "Error: No content returned from Gemini API. Possible safety block or API issue."

        except FileNotFoundError:
             return f"Error: Input file not found at {file_path}"
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            return f"Error: {e}"
        finally:
            if is_temp_image and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up temporary image: {file_path}")
                    temp_dir = os.path.dirname(file_path)
                    if not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {file_path}: {e}")
        
    def extract_more_info(self, file_path):
            prompt = '''
            Analyze the attached delivery note image and extract the following details.
            {"type": "text", "text": "Extract the following details from the delivery note:"},
            {"type": "text", "text": "Delivery Note Number"},
            {"type": "text", "text": "LPO Number"},
            {"type": "text", "text": "Driver Name"},
            {"type": "text", "text": "Driver Contact"},
            {"type": "text", "text": "Truck Number"},
            {"type": "text", "text": "Tank Number"},
            {"type": "text", "text": "Point of Origin"},
            {"type": "text", "text": "Final Destination"},
            {"type": "text", "text": "Port of Discharge"},
            {"type": "text", "text": "Country of Final Destination"},
            {"type": "text", "text": "1st Weight"},
            {"type": "text", "text": "2nd Weight"},
            {"type": "text", "text": "Net Weight"},
            {"type": "text", "text": "Products(Name, Quantity, Unit Price, Total Price, Batch Number, LPO Number)"},
            {"type": "text", "text": "OCR Seals"}
            
            Respond ONLY with the extracted information in the format listed above. Do not add any extra commentary before or after the list.
            Present the information clearly. If a field is not found, indicate 'Not Found'.'''

            is_temp_image = False
            try:
                if not os.path.exists(file_path):
                    return f"Error: Image file not found at {file_path}"

                print(f"Processing image for Gemini: {file_path}")
                img = PIL.Image.open(file_path)
            
                print("Sending request to Gemini API...")
                response = self.model.generate_content([prompt, img])

                print("Received response from Gemini.")
                if response.text:
                    try:
                        cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                        data = json.loads(cleaned_text)
                        if data:
                            return data
                        else:
                            return "Error: No content returned from Gemini API. Possible safety block or API issue."
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON response: {e}")
                        return "Error: No content returned from Gemini API. Possible safety block or API issue."
                else:
                    try:
                        print("Gemini response was empty. Checking feedback:", response.prompt_feedback)
                    except (AttributeError, IndexError):
                        pass
                    return "Error: No content returned from Gemini API. Possible safety block or API issue."

            except FileNotFoundError:
                return f"Error: Input file not found at {file_path}"
            except Exception as e:
                print(f"An unexpected error occurred during extraction: {e}")
                return f"Error: {e}"
            finally:
                if is_temp_image and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up temporary image: {file_path}")
                        temp_dir = os.path.dirname(file_path)
                        if not os.listdir(temp_dir):
                            os.rmdir(temp_dir)
                    except OSError as e:
                        print(f"Warning: Could not remove temporary file {file_path}: {e}")
        


def save_as_pdf(content, filename):
    pdf_filename = f"{filename}_extracted.pdf"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        if isinstance(content, dict):
            content = json.dumps(content, indent=2)

        lines = content.splitlines()
        for line in lines:
            try:
                pdf.multi_cell(0, 10, line)
                pdf.ln(1)
            except UnicodeEncodeError:
                print(f"Warning: Skipping line due to encoding issue with current font: {line[:50]}...")
                pdf.multi_cell(0, 5, txt="[Line skipped due to character encoding issue]")
                pdf.ln(1)

        pdf.output(pdf_filename)
        print(f"Content saved to {pdf_filename}")
    except Exception as e:
        print(f"Error saving content to PDF file {pdf_filename}: {e}")

def process_delivery_note(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None

    print(f"--- Starting processing for: {os.path.basename(file_path)} ---")
    try:
        extractor = DeliveryNoteTextExtractor()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return None
    except Exception as e:
        print(f"Initialization Error: {e}")
        return None

    content = extractor.extract_text(file_path)
    metadata = extractor.extract_more_info(file_path)

    if content and metadata:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        base_filename_meta_data = os.path.splitext(os.path.basename(file_path))[0] + '_meta_data'
        save_as_pdf(content, base_filename)
        save_as_pdf(metadata, base_filename_meta_data)
        print(f"--- Finished processing: {os.path.basename(file_path)} ---")
        return content
    else:
        print("\n--- Error during processing ---")
        print("-----------------------------\n")
        print(f"--- Failed processing: {os.path.basename(file_path)} ---")
        return None