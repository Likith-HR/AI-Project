from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from PIL import Image
import torch
from cnn_classifier import CNNWasteClassifier
import io
import os
import json
import datetime
import csv
from pathlib import Path

app = Flask(__name__)

# Set custom port
PORT = 8000  # Custom port

# Set the base URL prefix for all routes
BASE_URL = 'waste-classifier'

# Set the downloads directory
DOWNLOADS_DIR = str(Path.home() / "Downloads" / "waste-classifier-results")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the classifier
print(f"Using device: {device}")
print("Initializing waste classifier...")

try:
    # Make sure we're using the newly trained model
    classifier = CNNWasteClassifier(model_path='waste_cnn.pt', device=device)
    print(f"Classifier initialized with categories: {classifier.classes}")
except Exception as e:
    print(f"Error loading classifier: {e}")
    raise

@app.route('/')
def root():
    """Redirect root to the base URL"""
    return redirect(url_for('index'))

@app.route(f'/{BASE_URL}')
def index():
    return render_template('index.html', base_url=BASE_URL)

@app.route(f'/{BASE_URL}/minimize')
def minimize_waste():
    """Show tips for minimizing waste in daily life"""
    return render_template('minimize.html', base_url=BASE_URL)

@app.route(f'/{BASE_URL}/categories')
def categories():
    """Return the list of available waste categories"""
    return jsonify({
        'categories': classifier.classes
    })

@app.route(f'/{BASE_URL}/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get predictions
        predictions = classifier.classify(image)
        print(f"Raw predictions: {predictions}")
        
        # Get additional information for the top prediction
        top_prediction = predictions[0]
        class_name = top_prediction['type']
        print(f"Top prediction: {class_name}")
        
        class_info = classifier.get_class_info(class_name)
        
        # Prepare response
        response = {
            'predictions': predictions,
            'recycling_steps': class_info['recycling_steps'],
            'environmental_impact': class_info['environmental_impact'],
            'conservation_metrics': class_info['conservation_metrics']
        }
        
        print(f"Sending response: {json.dumps(response, indent=2)}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route(f'/{BASE_URL}/save-results', methods=['POST'])
def save_results():
    """Save conservation results to a CSV file in the Downloads folder"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        waste_type = data.get('waste_type')
        item_type = data.get('item_type')
        weight = data.get('weight')
        energy_saved = data.get('energy_saved')
        water_saved = data.get('water_saved')
        co2_reduced = data.get('co2_reduced')
        
        # Create a timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"waste_conservation_{timestamp}.csv"
        filepath = os.path.join(DOWNLOADS_DIR, filename)
        
        # Determine if this is a new file
        is_new_file = not os.path.exists(filepath)
        
        # Write to CSV file
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = ['Date', 'Waste Type', 'Item Type', 'Weight (kg)', 
                         'Energy Saved (kWh)', 'Water Saved (liters)', 'CO2 Reduced (kg)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only for new files
            if is_new_file:
                writer.writeheader()
                
            # Write data row
            writer.writerow({
                'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Waste Type': waste_type,
                'Item Type': item_type,
                'Weight (kg)': weight,
                'Energy Saved (kWh)': energy_saved,
                'Water Saved (liters)': water_saved,
                'CO2 Reduced (kg)': co2_reduced
            })
        
        return jsonify({
            'success': True,
            'message': 'Results saved successfully',
            'file_path': filepath
        })
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    access_url = f"http://localhost:{PORT}/{BASE_URL}"
    print(f"======================================================")
    print(f"Waste Classification System Starting")
    print(f"======================================================")
    print(f"Access the application at: {access_url}")
    print(f"Results will be saved to: {DOWNLOADS_DIR}")
    print(f"Press Ctrl+C to stop the server")
    print(f"======================================================")
    
    try:
        # Run with the custom port
        app.run(host='0.0.0.0', port=PORT, debug=True)
    except OSError as e:
        print(f"Error starting server on port {PORT}: {e}")
        print("Falling back to default port 5000...")
        fallback_url = f"http://localhost:5000/{BASE_URL}"
        print(f"Access the application at: {fallback_url}")
        app.run(host='0.0.0.0', port=5000, debug=True)
