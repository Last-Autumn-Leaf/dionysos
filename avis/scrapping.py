'''
# Librairies
import requests
import time

# URL de l'API de Google Maps
url = "https://maps.googleapis.com/maps/api/place/details/json"

# Paramètres de la requête
params = {
    "place_id": "ChIJA3t-Wk4ayUwRtilY5NR7LdE",  # ID de la localisation du restaurant
    "key": "AIzaSyC3Yqz2coWzEinLsjL_5EGTwOOOPOAFtWg"  # Remplacez par votre clé API
}

# Envoi de la requête à l'API
response = requests.get(url, params=params)

# Vérification du statut de la réponse
if response.status_code == 200:
    data = response.json()
    restaurant_name = data['result']['name']
    restaurant_adresse = data['result']['formatted_address']
    restaurant_rating = data['result']['rating']
    phone_number = data['result']['formatted_phone_number']

    print("Nom du restaurant:", restaurant_name)
    print("Adresse du restaurant:", restaurant_adresse)
    print("Note du restaurant:", restaurant_rating)
    print("Numéro de téléphone:", phone_number)

    # Récupérer les avis
    reviews = data['result']['reviews']

    # Vérifier si d'autres avis sont disponibles
    while 'next_page_token' in data:
        next_page_token = data['next_page_token']
        print(next_page_token)
        # Attendre quelques instants avant de faire une nouvelle requête avec le jeton de page suivante
        time.sleep(2)

        # Effectuer une nouvelle requête pour obtenir les avis de la page suivante
        params['pagetoken'] = next_page_token
        response = requests.get(url, params=params)
        data = response.json()

        # Ajouter les nouveaux avis à la liste existante
        reviews += data['result']['reviews']

    # Afficher tous les avis récupérés
    for review in reviews:
        author_name = review['author_name']
        rating = review['rating']
        review_text = review['text']

        print("\nAvis de", author_name)
        print("Note:", rating)
        print("Commentaire:", review_text)

else:
    print("Erreur lors de la requête à l'API.")
# Api : 
# Place ID: ChIJA3t-Wk4ayUwRtilY5NR7LdE
'''

from serpapi import GoogleSearch
params = {
  "engine": "google_maps_reviews",
  "data_id": "ChIJA3t:0xa4969e07ce3108de",
  "api_key": "f65a751358b015adbf1fa47da7a35bb4f9cc5e5821079838a6df3f1f4f32bf4f"
}

search = GoogleSearch(params)
results = search.get_dict()
print(results)
