import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import json
import joblib
import re
from PIL import Image
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import datetime

# Load the trained model
model_data = joblib.load('ckd_model.joblib')
model = model_data['model']  # Extract the actual model from the dictionary

# Set Tesseract path (update this path to match your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to extract text from an image file
def extract_text_from_image(image_path):
    try:
        # Open the image using PIL
        img = Image.open(image_path)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img)
        # append the text to a file
        # with open('output.txt', 'a') as file:
        #     file.write(text)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_all_fields():
    return {
        'hemo': {'label': 'Hemoglobin (g/dL)', 'required': True, 'type': 'numeric'},
        'wc': {'label': 'WBC Count (/cmm)', 'required': True, 'type': 'numeric'},
        'rc': {'label': 'RBC Count (Mil/cmm)', 'required': True, 'type': 'numeric'},
        'pcv': {'label': 'PCV (%)', 'required': True, 'type': 'numeric'},
        'bu': {'label': 'Blood Urea (mg/dL)', 'required': True, 'type': 'numeric'},
        'sc': {'label': 'Serum Creatinine (mg/dL)', 'required': True, 'type': 'numeric'},
        'sod': {'label': 'Sodium (mEq/L)', 'required': True, 'type': 'numeric'},
        'pot': {'label': 'Potassium (mmol/L)', 'required': True, 'type': 'numeric'},
        'sg': {'label': 'Specific Gravity', 'required': False, 'type': 'numeric'},
        'al': {'label': 'Albumin', 'required': False, 'type': 'categorical', 'options': [0, 1, 2, 3, 4, 5]},
        'su': {'label': 'Sugar', 'required': False, 'type': 'categorical', 'options': [0, 1, 2, 3, 4, 5]},
        'rbc': {'label': 'RBC', 'required': False, 'type': 'categorical', 'options': ['normal', 'abnormal']},
        'pc': {'label': 'Pus Cell', 'required': False, 'type': 'categorical', 'options': ['normal', 'abnormal']},
        'pcc': {'label': 'Pus Cell Clumps', 'required': False, 'type': 'categorical', 'options': ['present', 'notpresent']},
        'ba': {'label': 'Bacteria', 'required': False, 'type': 'categorical', 'options': ['present', 'notpresent']},
        'bgr': {'label': 'Blood Glucose Random (mg/dL)', 'required': False, 'type': 'numeric'},
        'bp': {'label': 'Blood Pressure (mm/Hg)', 'required': False, 'type': 'numeric'},
        'age': {'label': 'Age', 'required': False, 'type': 'numeric'},
        'dm': {'label': 'Diabetes Mellitus', 'required': False, 'type': 'categorical', 'options': ['yes', 'no']},
        'cad': {'label': 'Coronary Artery Disease', 'required': False, 'type': 'categorical', 'options': ['yes', 'no']},
        'appet': {'label': 'Appetite', 'required': False, 'type': 'categorical', 'options': ['good', 'poor']},
        'pe': {'label': 'Pedal Edema', 'required': False, 'type': 'categorical', 'options': ['yes', 'no']},
        'ane': {'label': 'Anemia', 'required': False, 'type': 'categorical', 'options': ['yes', 'no']},
        'htn': {'label': 'Hypertension', 'required': False, 'type': 'categorical', 'options': ['yes', 'no']}
    }

def estimate_missing_values(input_data):
    """Only estimate values that are actually missing, don't override existing values"""
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']
    estimated_values = input_data.copy()  # Start with existing values
    
    # Only fill in values that are actually missing
    for feature in feature_columns:
        if feature not in estimated_values or estimated_values[feature] is None:
            default_values = {
                'age': 45,
                'bp': 80,
                'sg': 1.020,
                'al': 0,
                'su': 0,
                'rbc': 1,
                'pc': 0,
                'pcc': 0,
                'ba': 0,
                'bgr': 100,
                'dm': 0,
                'cad': 0,
                'appet': 1,
                'pe': 0,
                'ane': 0,
                'htn': 0
            }
            estimated_values[feature] = default_values.get(feature, 0)
    
    return estimated_values

def prepare_input_for_model(input_data):
    """Prepare input data with correct data types for model prediction"""
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']
    
    # Create DataFrame with exact columns needed and in the correct order
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0.0
    
    # Map categorical values to numeric
    categorical_mapping = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0}
    }
    
    # Process each feature
    for feature in feature_columns:
        if feature in input_data:
            value = input_data[feature]
            
            # Handle categorical values
            if feature in categorical_mapping:
                if isinstance(value, str):
                    value = categorical_mapping[feature].get(value.lower(), 0)
            
            # Convert to float
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
            
            input_df.loc[0, feature] = value
    
    # Ensure all columns are float64
    input_df = input_df.astype(float)
    
    return input_df

def extract_patient_details(text):
    """Extract patient details and test values from the OCR text"""
    extracted_data = {
        'patient_details': {
            'name': None,
            'age': None,
            'gender': None,
            'patient_id': None
        },
        'test_values': {}
    }
    
    # Split text into lines for processing
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # Extract Patient Name
        if 'Patient Name' in line or 'patient Name' in line:
            name_parts = line.split(':')
            if len(name_parts) > 1:
                extracted_data['patient_details']['name'] = name_parts[1].strip()
        
        # Extract Age and Gender
        if 'Age / Gender' in line:
            try:
                # Extract age
                age_match = re.search(r'(\d+)\s*Y', line)
                if age_match:
                    extracted_data['patient_details']['age'] = int(age_match.group(1))
                
                # Extract gender
                if 'Male' in line:
                    extracted_data['patient_details']['gender'] = 'Male'
                elif 'Female' in line:
                    extracted_data['patient_details']['gender'] = 'Female'
            except Exception as e:
                st.warning(f"Error extracting age/gender: {str(e)}")
        
        # Extract Patient ID (UMR No)
        if 'UMR' in line:
            try:
                umr_match = re.search(r'UMR\d+', line)
                if umr_match:
                    extracted_data['patient_details']['patient_id'] = umr_match.group()
            except Exception as e:
                st.warning(f"Error extracting patient ID: {str(e)}")
        
        # Extract Test Values
        try:
            # Hemoglobin
            if 'Haemoglobin' in line and 'gm%' in line:
                match = re.search(r'(\d+\.?\d*)\s*13\.0\s*-\s*17\.0\s*gm%', line)
                if match:
                    extracted_data['test_values']['hemo'] = float(match.group(1))
            
            # WBC Count
            if 'T.WBC' in line:
                match = re.search(r'(\d+,?\d*)', line)
                if match:
                    wbc_value = float(match.group(1).replace(',', ''))
                    extracted_data['test_values']['wc'] = wbc_value
            
            # RBC Count
            if 'T.RBC' in line:
                match = re.search(r'(\d+\.?\d*)\s*4\.5-5\.5\s*Mil/cmm', line)
                if match:
                    extracted_data['test_values']['rc'] = float(match.group(1))
            
            # PCV
            if 'Haemotocrit(PCV)' in line:
                match = re.search(r'(\d+\.?\d*)\s*40\.0\s*-\s*50\.0', line)
                if match:
                    extracted_data['test_values']['pcv'] = float(match.group(1))
            
            # Blood Urea
            if 'Blood Urea Nitrogen' in line:
                match = re.search(r'(\d+\.?\d*)\s*6\.0\s*-\s*20\.0\s*mg/dl', line)
                if match:
                    extracted_data['test_values']['bu'] = float(match.group(1))
            
            # Serum Creatinine
            if 'Creatinine' in line:
                match = re.search(r'(\d+\.?\d*)\s*0\.6\s*-\s*1\.2\s*mg/dl', line)
                if match:
                    extracted_data['test_values']['sc'] = float(match.group(1))
            
            # Sodium
            if 'SODIUM' in line:
                match = re.search(r'(\d+\.?\d*)\s*136\s*-\s*146\s*mEq/L', line)
                if match:
                    extracted_data['test_values']['sod'] = float(match.group(1))
            
            # Potassium
            if 'Potassium' in line:
                match = re.search(r'(\d+\.?\d*)\s*3\.6\s*-\s*5\.1\s*mmol/L', line)
                if match:
                    extracted_data['test_values']['pot'] = float(match.group(1))
                    
        except Exception as e:
            st.warning(f"Error extracting test values: {str(e)}")
    
    # Validate extracted data
    st.write("Extracted Values:")
    st.write("Patient Details:", extracted_data['patient_details'])
    st.write("Test Values:", extracted_data['test_values'])
    
    return extracted_data

def process_images_with_status(uploaded_files):
    """Process uploaded images using image_extractor"""
    extracted_data = {
        'patient_details': {
            'name': None,
            'age': None,
            'gender': None,
            'patient_id': None
        },
        'test_values': {}
    }
    
    for uploaded_file in uploaded_files:
        with st.expander(f"Processing {uploaded_file.name}", expanded=True):
            try:
                # Display the image being processed
                img = Image.open(uploaded_file)
                st.image(img, caption=f"Processing: {uploaded_file.name}", width=300)
                
                # Show processing status
                with st.spinner('Extracting text from image...'):
                    extracted_text = extract_text_from_image(uploaded_file)
                    
                    if extracted_text:
                        st.success("Text extraction successful!")
                        st.write("Extracted Text Preview:")
                        st.text(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                        
                        # Parse the extracted text
                        with st.spinner('Parsing extracted text...'):
                            parsed_data = extract_patient_details(extracted_text)
                            if parsed_data:
                                # Merge data from multiple reports
                                for key in ['name', 'age', 'gender', 'patient_id']:
                                    if parsed_data['patient_details'].get(key):
                                        extracted_data['patient_details'][key] = parsed_data['patient_details'][key]
                                
                                # Merge test values
                                extracted_data['test_values'].update(parsed_data.get('test_values', {}))
                                st.success("Data extraction complete!")
                            else:
                                st.warning("No data could be extracted from this image")
                    else:
                        st.error("No text could be extracted from this image")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
    
    return extracted_data

def generate_pdf_report(patient_details, input_data, prediction, probability):
    """Generate a PDF report using reportlab"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont('Helvetica-Bold', 16)
    c.drawString(30, height - 50, 'CKD Prediction Report')
    c.setFont('Helvetica', 10)
    c.drawString(30, height - 70, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Prediction Result
    c.setFont('Helvetica-Bold', 14)
    result_text = 'CKD Detected' if prediction[0] == 1 else 'No CKD Detected'
    c.drawString(30, height - 100, result_text)
    c.setFont('Helvetica', 12)
    c.drawString(30, height - 120, f'Confidence: {probability[0][1]:.1%}')
    
    # Patient Information
    y_position = height - 160
    c.setFont('Helvetica-Bold', 12)
    c.drawString(30, y_position, 'Patient Information')
    c.setFont('Helvetica', 10)
    y_position -= 20
    
    for key, value in patient_details.items():
        c.drawString(30, y_position, f'{key.title()}: {value}')
        y_position -= 15
    
    # Test Results
    y_position -= 20
    c.setFont('Helvetica-Bold', 12)
    c.drawString(30, y_position, 'Test Results')
    c.setFont('Helvetica', 10)
    y_position -= 20
    
    all_fields = get_all_fields()
    for key, value in input_data.items():
        if key in all_fields:
            label = all_fields[key]['label']
            if y_position < 50:  # Check if we need a new page
                c.showPage()
                y_position = height - 50
                c.setFont('Helvetica', 10)
            c.drawString(30, y_position, f'{label}: {value}')
            y_position -= 15
    
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

def display_results(patient_details, input_data, prediction, probability):
    """Display prediction results in a professional format"""
    # Create three columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <style>
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .prediction-positive {
            background-color: #ff4b4b;
            color: white;
        }
        .prediction-negative {
            background-color: #00cc00;
            color: white;
        }
        .report-section {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .patient-info {
            display: grid;
            grid-template-columns: auto auto;
            gap: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Prediction Result
        prediction_class = "prediction-positive" if prediction[0] == 1 else "prediction-negative"
        st.markdown(f"""
        <div class="prediction-box {prediction_class}">
            <h2>{'CKD Detected' if prediction[0] == 1 else 'No CKD Detected'}</h2>
            <h3>Confidence: {probability[0][1]:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Patient Information
        st.markdown("""
        <div class="report-section">
            <h3>Patient Information</h3>
            <div class="patient-info">
        """, unsafe_allow_html=True)
        
        for key, value in patient_details.items():
            st.markdown(f"<p><strong>{key.title()}:</strong> {value}</p>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Test Results
        st.markdown("""
        <div class="report-section">
            <h3>Test Results</h3>
            <div class="patient-info">
        """, unsafe_allow_html=True)
        
        all_fields = get_all_fields()
        for key, value in input_data.items():
            if key in all_fields:
                label = all_fields[key]['label']
                st.markdown(f"<p><strong>{label}:</strong> {value}</p>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Download Report Button
        try:
            pdf_data = generate_pdf_report(patient_details, input_data, prediction, probability)
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_data,
                file_name=f"ckd_report_{patient_details['patient_id']}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

def display_form_fields(extracted_data, all_fields):
    """Display form fields with extracted data and allow manual input"""
    with st.form("patient_form"):
        # Patient Details Section
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(
                "Patient Name *", 
                value=extracted_data['patient_details'].get('name', ''),
                key="name"
            )
            age = st.number_input(
                "Age *", 
                value=extracted_data['patient_details'].get('age', 0),
                min_value=0,
                max_value=120,
                key="age"
            )
        with col2:
            gender = st.selectbox(
                "Gender *",
                ["Male", "Female"],
                index=0 if extracted_data['patient_details'].get('gender') == "Male" else 1,
                key="gender"
            )
            patient_id = st.text_input(
                "Patient ID *",
                value=extracted_data['patient_details'].get('patient_id', ''),
                key="patient_id"
            )

        # Test Values Section
        st.subheader("Test Values")
        col1, col2 = st.columns(2)
        input_data = {}
        
        for i, (key, field_info) in enumerate(all_fields.items()):
            with col1 if i % 2 == 0 else col2:
                extracted_value = extracted_data['test_values'].get(key, None)
                
                if field_info['type'] == 'numeric':
                    default_value = 0.0
                    if extracted_value is not None:
                        try:
                            default_value = float(extracted_value)
                        except (ValueError, TypeError):
                            default_value = 0.0
                            
                    value = st.number_input(
                        f"{field_info['label']} {'*' if field_info['required'] else ''}",
                        value=default_value,
                        format="%.2f",
                        key=f"input_{key}",
                        help=("Required field" if field_info['required'] else "Optional field")
                    )
                    if value != 0.0 or field_info['required']:
                        input_data[key] = value
                
                elif field_info['type'] == 'categorical':
                    options = field_info['options']
                    default_index = 0
                    if extracted_value is not None:
                        try:
                            default_index = options.index(extracted_value)
                        except ValueError:
                            default_index = 0
                    
                    value = st.selectbox(
                        f"{field_info['label']} {'*' if field_info['required'] else ''}",
                        options=options,
                        index=default_index,
                        key=f"input_{key}",
                        help=("Required field" if field_info['required'] else "Optional field")
                    )
                    input_data[key] = value

        # Submit button
        submitted = st.form_submit_button("Verify and Predict")
        
        if submitted:
            # Collect patient details
            patient_details = {
                'name': name,
                'age': age,
                'gender': gender,
                'patient_id': patient_id
            }
            
            # Validate the data
            form_valid, validation_messages = validate_input_data(patient_details, input_data, all_fields)
            
            # Store validation results in session state
            st.session_state.form_valid = form_valid
            st.session_state.validation_messages = validation_messages
            
            # Show validation messages
            for msg in validation_messages:
                st.warning(msg)
            
            return patient_details, input_data, form_valid, submitted
        
        return (
            {'name': name, 'age': age, 'gender': gender, 'patient_id': patient_id},
            input_data,
            False,  # Not valid until submitted and validated
            submitted
        )

def validate_input_data(patient_details, input_data, all_fields):
    """Validate both OCR-extracted and manually entered data"""
    validation_messages = []
    form_valid = True
    
    # Validate patient details
    if not all([
        patient_details.get('name'), 
        patient_details.get('age'), 
        patient_details.get('gender'), 
        patient_details.get('patient_id')
    ]):
        validation_messages.append("❌ All patient details are required")
        form_valid = False
    
    # Validate test values
    required_fields = {k: v for k, v in all_fields.items() if v['required']}
    missing_required = []
    invalid_values = []
    
    for key, field_info in required_fields.items():
        value = input_data.get(key)
        if value is None or value == 0.0:
            missing_required.append(field_info['label'])
        else:
            # Validate value ranges based on field
            try:
                value = float(value)
                # Add field-specific validation ranges
                if key == 'hemo' and (value < 0 or value > 20):
                    invalid_values.append(f"{field_info['label']}: Value should be between 0-20")
                elif key == 'wc' and (value < 0 or value > 50000):
                    invalid_values.append(f"{field_info['label']}: Value should be between 0-50000")
                # Add more field-specific validations as needed
            except ValueError:
                invalid_values.append(f"{field_info['label']}: Invalid numeric value")
    
    if missing_required:
        validation_messages.append(f"❌ Required test values missing: {', '.join(missing_required)}")
        form_valid = False
    
    if invalid_values:
        validation_messages.append(f"❌ Invalid values detected: {', '.join(invalid_values)}")
        form_valid = False
    
    return form_valid, validation_messages

def main():
    st.title("CKD Prediction System")
    
    input_method = st.radio("Choose input method:", ["Manual Input", "Image Upload"])
    
    all_fields = get_all_fields()
    input_data = {}
    patient_details = {
        'name': None,
        'age': None,
        'gender': None,
        'patient_id': None
    }

    if input_method == "Manual Input":
        with st.form("manual_input_form"):
            st.subheader("Patient Details")
            col1, col2 = st.columns(2)
            with col1:
                patient_details['name'] = st.text_input("Patient Name")
                patient_details['age'] = st.number_input("Age", min_value=0, max_value=120)
            with col2:
                patient_details['gender'] = st.selectbox("Gender", ["Male", "Female"])
                patient_details['patient_id'] = st.text_input("Patient ID")

            st.subheader("Test Values")
            st.info("Fields marked with * are required for prediction")
            
            st.markdown("### Blood Tests")
            col1, col2 = st.columns(2)
            blood_tests = ['hemo', 'wc', 'rc', 'pcv', 'bgr', 'bu', 'sc', 'sod', 'pot']
            
            for i, key in enumerate(blood_tests):
                with col1 if i % 2 == 0 else col2:
                    field_info = all_fields[key]
                    value = st.number_input(
                        field_info['label'],
                        value=0.0,
                        format="%.2f",
                        help="This field is required" if field_info['required'] else "Optional field"
                    )
                    if value != 0.0 or field_info['required']:
                        input_data[key] = value

            form_valid = True
            validation_messages = []
            
            if not all([patient_details['name'], patient_details['age'], patient_details['gender'], patient_details['patient_id']]):
                validation_messages.append("❌ All patient details are required")
                form_valid = False
            
            required_fields = {k: v for k, v in all_fields.items() if v['required']}
            missing_required = [
                all_fields[k]['label'] 
                for k in required_fields 
                if k not in input_data or input_data[k] == 0.0
            ]
            
            if missing_required:
                validation_messages.append(f"❌ Required test values missing: {', '.join(missing_required)}")
                form_valid = False
            
            for msg in validation_messages:
                st.warning(msg)
            
            submitted = st.form_submit_button(
                "Verify and Predict",
                disabled=not form_valid,
                help="Please fill all required fields to enable prediction"
            )
            
            if submitted and form_valid:
                with st.spinner('Making prediction...'):
                    complete_values = estimate_missing_values(input_data)
                    input_df = prepare_input_for_model(complete_values)
                    
                    prediction = model.predict(input_df)
                    probability = model.predict_proba(input_df)
                    
                    display_results(patient_details, complete_values, prediction, probability)
    
    else:
        uploaded_files = st.file_uploader(
            "Upload patient report images", 
            type=['jpg', 'png', 'jpeg'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner('Processing uploaded reports...'):
                extracted_data = process_images_with_status(uploaded_files)
            
            # Display form and get results
            patient_details, input_data, form_valid, submitted = display_form_fields(extracted_data, all_fields)
            
            if submitted and form_valid:
                with st.spinner('Making prediction...'):
                    complete_values = estimate_missing_values(input_data)
                    input_df = prepare_input_for_model(complete_values)
                    prediction = model.predict(input_df)
                    probability = model.predict_proba(input_df)
                    display_results(patient_details, complete_values, prediction, probability)

if __name__ == "__main__":
    main() 