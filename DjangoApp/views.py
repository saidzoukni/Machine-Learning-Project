from django.shortcuts import render
import os
import joblib
# Create your views here.
def index(request):	
    return render(request, 'index.html')

def regLog_details(request):
    return render(request, 'regLog_details.html')

def regLog_atelier(request):
    return render(request, 'regLog_atelier.html')

def regLog_tester(request):
    return render(request, 'vehicles_form.html')

def load_models(name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models_ai')
    model_path = os.path.join(models_dir, name)
    # Try joblib first; fall back to pickle if joblib is unavailable
    try:
        import joblib  # type: ignore
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def regLog_prediction(request):
    if request.method == 'POST':
        hauteur = float(request.POST.get('hauteur'))
        nbr_roues = float(request.POST.get('Nombre_de_roues'))

        model = load_models('logreg_model.pkl')

        prediction = model.predict([[hauteur, nbr_roues]])
        predicted_class = prediction[0]

        type_vehicules = {0: 'Camion', 1: 'Touristique'}
        img_url = {'Camion': 'images/camion.jpg', 'Touristique': 'images/touristique.jpg'}
        pred_vehicle = type_vehicules[predicted_class]
        pred_img = img_url[pred_vehicle]

        input_data = {
            'hauteur': hauteur,
            'nbr_roues': nbr_roues
        }

        context = {
            'type_vehicule': pred_vehicle,
            'img_vehicule': pred_img,
            'initial_data': input_data
        }
        return render(request, 'regLog_results.html', context)
    return render(request, 'vehicles_form.html')

def dectree_details(request):
    return render(request, 'dectree_details.html')

def dectree_atelier(request):
    return render(request, 'dectree_atelier.html')

def dectree_tester(request):
    return render(request, 'restaurant_form_dectree.html')

def dectree_prediction(request):
    if request.method == 'POST':
        # Get form data
        name = request.POST.get('name', '')
        rest_type = request.POST.get('rest_type', '')
        city = request.POST.get('city', '')
        type_service = request.POST.get('type', '')
        rate = float(request.POST.get('rate', 4.0))
        cost = float(request.POST.get('cost', 500))
        cuisines = request.POST.get('cuisines', '')
        dish_liked = request.POST.get('dish_liked', '')
        book_table = request.POST.get('book_table', 'No')

        # Load model
        model = load_models('dectree_model.pkl')
        
        # Try to load encoders and scaler if they exist
        # Note: You need to save LabelEncoders and StandardScaler during model training
        try:
            label_encoders = load_models('label_encoders.pkl')
            scaler = load_models('scaler.pkl')
            
            # Encode categorical features
            name_encoded = label_encoders['name'].transform([name])[0] if 'name' in label_encoders else 0
            rest_type_encoded = label_encoders['rest_type'].transform([rest_type])[0] if 'rest_type' in label_encoders else 0
            city_encoded = label_encoders['city'].transform([city])[0] if 'city' in label_encoders else 0
            type_encoded = label_encoders['type'].transform([type_service])[0] if 'type' in label_encoders else 0
            cuisines_encoded = label_encoders['cuisines'].transform([cuisines])[0] if 'cuisines' in label_encoders else 0
            dish_liked_encoded = label_encoders['dish_liked'].transform([dish_liked])[0] if 'dish_liked' in label_encoders else 0
            book_table_encoded = label_encoders['book_table'].transform([book_table])[0] if 'book_table' in label_encoders else (1 if book_table == 'Yes' else 0)
            
            # Scale numerical features
            rate_scaled, cost_scaled = scaler.transform([[rate, cost]])[0]
            
            # Prepare feature array in the same order as training: name, rest_type, dish_liked, cuisines, type, city, rate, cost, book_table
            features = [[name_encoded, rest_type_encoded, dish_liked_encoded, cuisines_encoded, 
                        type_encoded, city_encoded, rate_scaled, cost_scaled, book_table_encoded]]
        except:
            # Fallback: use simple encoding if encoders not available
            # This is a simplified version - you should save and load the actual encoders
            import hashlib
            name_encoded = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10000
            rest_type_encoded = int(hashlib.md5(rest_type.encode()).hexdigest()[:8], 16) % 100
            city_encoded = int(hashlib.md5(city.encode()).hexdigest()[:8], 16) % 1000
            type_encoded = int(hashlib.md5(type_service.encode()).hexdigest()[:8], 16) % 100
            cuisines_encoded = int(hashlib.md5(cuisines.encode()).hexdigest()[:8], 16) % 1000
            dish_liked_encoded = int(hashlib.md5(dish_liked.encode()).hexdigest()[:8], 16) % 1000
            book_table_encoded = 1 if book_table == 'Yes' else 0
            # Simple scaling (you should use the actual StandardScaler)
            rate_scaled = (rate - 2.5) / 2.5
            cost_scaled = (cost - 500) / 500
            
            features = [[name_encoded, rest_type_encoded, dish_liked_encoded, cuisines_encoded, 
                        type_encoded, city_encoded, rate_scaled, cost_scaled, book_table_encoded]]

        # Make prediction
        prediction = model.predict(features)
        predicted_class = prediction[0]

        # Map prediction to readable format
        online_order_result = 'Oui' if predicted_class == 1 else 'Non'
        online_order_english = 'Yes' if predicted_class == 1 else 'No'

        input_data = {
            'name': name,
            'rest_type': rest_type,
            'city': city,
            'type': type_service,
            'rate': rate,
            'cost': cost,
            'cuisines': cuisines,
            'dish_liked': dish_liked,
            'book_table': book_table
        }

        context = {
            'online_order': online_order_result,
            'online_order_english': online_order_english,
            'initial_data': input_data
        }
        return render(request, 'dectree_results.html', context)
    return render(request, 'restaurant_form_dectree.html')