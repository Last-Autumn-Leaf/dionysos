# Librairies
import os
# Scrapping
from serpapi import GoogleSearch

# Gerer les données
import pandas as pd

#Gerer les dates
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime
import locale
# Définir la langue française
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Gerer les logs
import logging as lg
lg.basicConfig(level=lg.INFO)



class serApi(): 
    '''
    Cette classe permet de récupérer les avis d'un lieu sur Google Maps
    '''
    @staticmethod
    def get_api_key(provider,chemin = os.path.join( 'api_key.txt')):
        """Récupère la clé d'API dans le fichier api_key.txt pour le fournisseur spécifié"""
        with open(chemin, 'r') as f:
            api_keys = f.readlines()

        for api_key in api_keys:
            if api_key.startswith(provider):
                return api_key.split(':')[1].strip()
        # Si aucune clé n'a été trouvée pour le fournisseur spécifié
        return None

    @staticmethod
    def get_data_id(query = '150 Sainte-Catherine O local 201C, Montréal, QC H5B 1B3'):
        """
        Récupère l'identifiant du lieu 

        Input:
        - query : adresse du lieu

        Output:
        - data_id : identifiant du lieu
        - latitude : latitude du lieu
        - longitude : longitude du lieu
        """
        api_key = serApi.get_api_key('SerpApi')
        
        # Definition des paramètres de la requête
        params = {
        "engine": "google_maps",
        "q": query,
        #'ll' = "@{},{}z".format(latitude, longitude),
        "type": "search",
        "api_key": api_key
        }
        # On effectue la requête
        search = GoogleSearch(params)
        results = search.get_dict()
        local_results = results["place_results"]
        # On liste les données à recuperer
        data_id = local_results["data_id"]
        latitude = local_results["gps_coordinates"]["latitude"]
        longitude = local_results["gps_coordinates"]["longitude"]
        return data_id,latitude,longitude

    def get_key_data(query = '150 Sainte-Catherine O local 201C, Montréal, QC H5B 1B3'):
        """
        Récupère les données d'un lieu

        Input:
        - query : adresse du lieu

        Output:
        - data_id : identifiant du lieu
        - nb_review : nombre de review du lieu
        """

        api_key = serApi.get_api_key('SerpApi')
        data_id,latitude,longitude = serApi.get_data_id(query = query)

        place_id = "!4m5!3m4!1s" + str(data_id) + "!8m2!3d" + str(latitude) + "!4d" + str(longitude)
        # Definition des paramètres de la requête
        params = {
        "engine": "google_maps",
        "type": "place",
        "place_id": place_id,
        "api_key": api_key
        }

        # On effectue la requête
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        place_results = results["place_results"]

        # On liste les données à recuperer
        nom = place_results["title"]
        note = place_results["rating"]
        nb_review = place_results["reviews"]
        type_lieu = place_results["type"]
        data_id = place_results["data_id"]
        adresse = place_results["address"]

        # On affiche les données
        print("Nom : {}".format(nom))
        print("Adresse : {}".format(adresse))
        print("Note : {}".format(note))
        print("Nombre de review : {}".format(nb_review))
        print("Type de lieu : {}".format(type_lieu))

        return data_id, nb_review

class pre_process():

    def convert_relative_date(relative_date):

        '''
        Cette fonction convertit une date relative en date absolue
        '''
        if ("month" in relative_date) or ('months' in relative_date):
            if relative_date.split()[0] == "a":
                return (datetime.now() - relativedelta(months=1)).strftime("%Y %m %d")
            else:
                months_ago = int(relative_date.split()[0])
                return (datetime.now() - relativedelta(months=months_ago)).strftime("%Y %m %d")
        
        elif ("week" in relative_date) or ("weeks" in relative_date):
            if relative_date.split()[0] == "a":
                return (datetime.now() - relativedelta(weeks=1)).strftime("%Y %m %d")
            else:
                weeks_ago = int(relative_date.split()[0])
                return (datetime.now() - relativedelta(weeks=weeks_ago)).strftime("%Y %m %d")
        
        elif ("year" in relative_date) or ("years" in relative_date):
            if relative_date.split()[0] == "a":
                return (datetime.now() - relativedelta(years=1)).strftime("%Y %m %d")
            else:
                years_ago = int(relative_date.split()[0])
                return (datetime.now() - relativedelta(years=years_ago)).strftime("%Y %m %d")
        
        elif ("day" in relative_date) or ("days" in relative_date):
            if relative_date.split()[0] == "a":
                return (datetime.now() - relativedelta(days=1)).strftime("%Y %m %d")
            else:
                days_ago = int(relative_date.split()[0])
                return (datetime.now() - relativedelta(days=days_ago)).strftime("%Y %m %d")

        # Par défaut, utilisez le parseur de date de `dateutil`
        return parse(relative_date).strftime("%y %m %d")

    def extract_translated_text(text):
        '''
        Cette fonction permet d'extraire le texte traduit d'un avis
        '''
        start_tag = "(Translated by Google)"
        end_tag = "(Original)"
        
        start_index = text.find(start_tag)
        end_index = text.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            return text[start_index + len(start_tag):end_index].strip()
        else:
            return text
        
    def scrap_review():
        api_key = serApi.get_api_key('SerpApi') 
        data_id = serApi.get_data_id()

        # Definition des paramètres de la requête
        params = {
        "engine": "google_maps_reviews",
        "data_id": data_id,
        "api_key": api_key
        }

        # On récupère les avis
        avis = {'nom':[], 'note':[], 'date':[], 'texte':[]}
        num_reviews = nb_review  # Nombre total d'avis souhaité
        i = 0
        while i < num_reviews:
            # On récupère les avis par paquet de 10
            lg.info("Récupération des avis {}/{}".format(i, num_reviews))
            # On passe aux pages suivantes
            if i !=0 :
                params['next_page_token'] = next_page_token
            # On effectue la requête
            search = GoogleSearch(params)
            results = search.get_dict()
            # On récupère le token de la page suivante
            next_page_token = results['serpapi_pagination']['next_page_token']
            # Vérifiez si les avis sont présents dans les résultats
            if 'reviews' in results:
                reviews = results['reviews']
                # On ajoute les avis au dictionnaire
                for review in reviews:
                    avis['nom'].append(review['user']['name'])
                    avis['note'].append(review['rating'])
                    avis['date'].append(review['date'])
                    avis['texte'].append(review['snippet'])
            # On incrémente le compteur
            i += 10

        # On transforme le dictionnaire en dataframe
        df_avis = pd.DataFrame(avis)
        # On sauvegarde le dataframe en csv
        
        # On convertit la colonne 'date' en objet datetime
        df_avis['date'] = df_avis['date'].apply(pre_process.convert_relative_date)
        df_avis['texte'] = df_avis['texte'].apply(pre_process.extract_translated_text)

        # on classe les avis par date
        df_avis = df_avis.sort_values(by='date', ascending=False)
        
        df_avis.to_csv('avis/data.csv', index=False)

        # Vérifiez si aucun avis n'a été trouvé
        if len(df_avis) == 0:
            print("Aucun avis n'a été trouvé")

if __name__ == '__main__':
    serApi.get_key_data()