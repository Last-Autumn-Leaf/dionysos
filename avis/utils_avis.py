# Librairies
import os
# Scrapping
from serpapi import GoogleSearch

# Gerer les données
import pandas as pd

#Gerer les dates
import time
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime
import calendar
import locale
# Définir la langue française
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Gerer latex
from pylatex import Document, Section, Subsection, Command, Center
from pylatex.utils import NoEscape, bold, italic, escape_latex

# GPT
import openai

# analyse des sentimennts 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Gerer les logs
import logging as lg
lg.basicConfig(level=lg.INFO)

class scrap(): 
    '''
    Cette classe permet de récupérer les avis d'un lieu sur Google Maps
    '''

    def get_api_key(provider,chemin = os.path.join('avis', 'api_key.txt')):
        """Récupère la clé d'API dans le fichier api_key.txt pour le fournisseur spécifié"""
        with open(chemin, 'r') as f:
            api_keys = f.readlines()

        for api_key in api_keys:
            if api_key.startswith(provider):
                return api_key.split(':')[1].strip()
        # Si aucune clé n'a été trouvée pour le fournisseur spécifié
        return None

    def get_data_id():
        """
        Récupère l'identifiant du lieu 
        """
        api_key = scrap.get_api_key('SerpApi')

        # Definition des paramètres de la requête
        params = {
        "engine": "google_maps",
        "type": "place",
        "data": "!3m1!5s0x4cc91a45a1cf65b7:0x3310f1891dd11e56!4m8!3m7!1s0x4cc91a4e5a7e7b03:0xd12d7bd4e45829b6!8m2!3d45.5076517!4d-73.5653556!9m1!1b1!16s%2Fg%2F11c1_w0s1b",
        "api_key": api_key
        }

        # On effectue la requête
        search = GoogleSearch(params)
        results = search.get_dict()
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


    def convert_relative_date(relative_date):
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
        start_tag = "(Translated by Google)"
        end_tag = "(Original)"
        
        start_index = text.find(start_tag)
        end_index = text.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            return text[start_index + len(start_tag):end_index].strip()
        else:
            return text
        
    def scrap_review():
        api_key = scrap.get_api_key('SerpApi') 
        data_id, nb_review = scrap.get_data_id()

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
        df_avis['date'] = df_avis['date'].apply(scrap.convert_relative_date)
        df_avis['texte'] = df_avis['texte'].apply(scrap.extract_translated_text)

        # on classe les avis par date
        df_avis = df_avis.sort_values(by='date', ascending=False)
        
        df_avis.to_csv('avis/data.csv', index=False)

        # Vérifiez si aucun avis n'a été trouvé
        if len(df_avis) == 0:
            print("Aucun avis n'a été trouvé")

class rapport():

    # ecris une fonction qui donne le nombre de review, la moyenne et la progression (par rapport au mois passé) pour un mois donné
    def get_review_month(nom_mois):
        '''
        Fonction qui donne le nombre de review, la moyenne et la progression (par rapport au mois passé) pour un mois donné
        '''

        nb_mois = list(calendar.month_name).index(nom_mois)

        # on récupère le dataframe
        df = pd.read_csv('avis/data.csv')
        # on convertit la colonne 'date' en objet datetime
        df['date'] = pd.to_datetime(df['date'])
        # on filtre les avis du mois passé
        df_mois = df[df['date'].dt.month == nb_mois]
        # on compte le nombre d'avis du mois ou le texte est non vide
        nb_avis = df_mois[df_mois['texte'].apply(str) != 'nan'].shape[0]
        # on calcule la moyenne du mois 
        moyenne = round(df_mois['note'].mean(),2)

        # On calcule la moyenne du mois passé
        try : 
            df_mois_old = df[df['date'].dt.month == (nb_mois - 1)]
            moyenne_old = df_mois_old['note'].mean()

            # on calcule la progression du mois par rapport au mois passé
            progression = round((moyenne - moyenne_old) / moyenne_old * 100, 2)
        except :
            progression = 0
        
        # log
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] les données sur le nombre d'avis, la moyenne des notes et la progression du mois ont été récupérées")
        return nb_avis, moyenne, progression
    
    def get_rep_review_month(nom_mois,chemin_fichier_csv = 'avis/data.csv'):

        nb_mois = list(calendar.month_name).index(nom_mois)
        # on récupère le dataframe
        df = pd.read_csv(chemin_fichier_csv)
        # on convertit la colonne 'date' en objet datetime
        df['date'] = pd.to_datetime(df['date'])
        # on filtre les avis du mois passé
        df_mois = df[df['date'].dt.month == nb_mois]

        # on récupère les avis
        review = df_mois['texte'].tolist()
        reponse_avis = ""
        j = 0
        for i in range(len(review)):
            try:
                rep = gpt.generate_response(review[i])
                reponse_avis += ("{}. Avis: {}".format(j+1, review[i]))
                reponse_avis += "\n"
                reponse_avis += "-   Réponse : {}".format(rep)    
                reponse_avis += "\n\n"
                j+=1
            except: 
                continue
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Les réponses aux avis ont été générées")
        return reponse_avis

        
    def create_latex(nom_mois,old_sum,nom='La Cage',chemin_fichier_csv ='avis/data.csv'):
        
        '''
        Cette fonction crée un document LaTeX qui contient le rapport des avis d'un mois donné
        '''
        # recupere le fichier csv avis/data.csv
        # recuperation des données
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de la création du rapport")
        nb_avis, moyenne, progression = rapport.get_review_month(nom_mois)
        nb_positif, nb_negatif, nb_neutre = feeling.nb_sentim(nom_mois)
        # recuperation des données sur les points d'amélioration
        summary, improvement_points, improvement_observationsmary = gpt.generate_report(chemin_fichier_csv, nom_mois, old_sum)
        # recuperation des réponses aux avis
        reponse_avis = rapport.get_rep_review_month(nom_mois,chemin_fichier_csv = 'avis/data.csv')
        # Création du document LaTeX
        doc = Document('rapport')
        titre = "Rapport d'analyse des avis de la Cage pour le mois de {}".format(nom_mois)
        # Informations générales
        doc.append(Command('begin', 'center'))
        doc.append(Command('textbf',Command('Large',titre)))
        doc.append(Command("\ "))
        doc.append(Command('vspace', '0.5cm'))
        doc.append(Command('today'))
        doc.append(Command('end', 'center'))

        doc.append(Command('vspace', '1cm'))
        
        # Page de titre
        with doc.create(Section('Analyse des tendances')):
            doc.append(Command('begin', 'itemize'))
            doc.append(Command('item', 'Pour ce mois, on retrouve {} avis dont {} avis positif ({}%), {} avis négatif ({}%) et {} avis neutre ({}%)  '.format(nb_avis,nb_positif,round(nb_positif/nb_avis*100,2),nb_negatif,round(nb_negatif/nb_avis*100,2),nb_neutre,round(nb_neutre/nb_avis*100,2))))
            doc.append(Command('item', 'Note moyenne : {}/5'.format(moyenne)))
            if str(progression) != 'nan' : 
                doc.append(Command('begin', 'itemize'))
                if progression > 0:
                    doc.append(Command('item', 'La moyenne des avis est en progression de {}% par rapport au mois passé'.format(progression)))
                elif progression < 0:
                    doc.append(Command('item', 'La moyenne des avis est en régression de {}% par rapport au mois passé'.format(progression)))
                else:
                    doc.append(Command('item', 'La moyenne des avis est stable par rapport au mois passé'))
                doc.append(Command('end', 'itemize'))   
            doc.append(Command('end', 'itemize'))

        # Résumé du mois
        with doc.create(Section('Résumé du mois')):
            # Section résumé des avis
            doc.append(summary)
            doc.append(Command('vspace', '0.5cm'))

        # Section Sugestions et Ameliorations
        with doc.create(Section('Points clés à améliorer')):
            doc.append('Les points clé à améliorer pour ce mois sont:')
            # Conversion en liste d'items
            items = improvement_points.split('\n')
            items = [item.strip().lstrip('- ') for item in items if item.strip()]

            # Génération du code LaTeX
            doc.append(Command('begin', 'itemize'))
            for item in items:
                doc.append(Command('item', item))
            doc.append(Command('end', 'itemize'))
            doc.append(Command('vspace', '0.5cm'))
        with doc.create(Section('Améliorations observées par rapport aux mois passés')):
            doc.append(improvement_observationsmary)

        # Feuille des réponses aux avis
        with doc.create(Section('Réponses aux avis')):
            doc.append(reponse_avis)

        try :
            # Générer le fichier PDF du rapport
            doc.generate_pdf('avis/rapport/{}'.format(nom_mois), clean_tex=True)
        except :
            pass

        return summary
    def clean_file(chemin = 'avis/rapport/'):
        # Supprimer les fichiers temporaires du répertoire courant
        for filename in os.listdir("avis/rapport"):
            if filename.endswith(('.aux', '.log', '.out', '.synctex.gz', '.tex','.fdb_latexmk','.fls')):
                os.remove(chemin + filename)

    def multi_rap(mois):
        all_sum = [ 'pas de résumé pour le mois précédent']
        for nom_mois in mois:
            print("Écriture du rapport pour le mois de {}".format(nom_mois))
            try : 
                # Récupération de la clé API OpenAI
                sum = rapport.create_latex(nom_mois,old_sum=all_sum[-1],nom='La Cage',chemin_fichier_csv ='avis/data.csv')
                all_sum.append(sum)
            except :
                pass
        time.sleep(5)
        rapport.clean_file()
        return all_sum

class gpt():
    openai.api_key = scrap.get_api_key('OPENAI_API_KEY')

    def generate_summary(avis,mois):
        # Paramètres de la génération de texte avec GPT
        model = "gpt-3.5-turbo"
        max_tokens = 2500

        # Génération du résumé des avis du mois
        prompt = "Fais moi un résumé en français des avis du mois de {} :".format(mois)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": avis}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        summary = response.choices[0].message.content.strip()
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le résumé des avis a été généré")
        return summary

    def generate_improvement_points(avis,mois):
        # Paramètres de la génération de texte avec GPT
        model = "gpt-3.5-turbo"
        max_tokens = 2500

        # Génération des propositions de points d'amélioration
        prompt = "Liste des points d'améliorations à faire pour le mois de {} (En francais) :".format(mois)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": avis}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        improvement_points = response.choices[0].message.content.strip()
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] les points d'amélioration ont été générés")
        return improvement_points

    def sumary2sum(sum):
        # Paramètres de la génération de texte avec GPT
        model = "gpt-3.5-turbo"
        max_tokens = 100

        # Génération du résumé des avis du mois
        prompt = "Fais moi un résumé très court (100 mot max) de ce texte :"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sum}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        summary = response.choices[0].message.content.strip()
        return summary
    def generate_improvement_observations(sum,old_sum,mois):
        # Paramètres de la génération de texte avec GPT
        model = "gpt-3.5-turbo"
        max_tokens = 1000
        sum = gpt.sumary2sum(sum)
        old_sum = gpt.sumary2sum(old_sum)
        texte = 'résumé du mois : ' + sum +'résumé du mois avant: ' +old_sum
        # Génération des observations d'amélioration par rapport aux mois précédents
        prompt = "Observations d'amélioration en francais pour le mois de {} par rapport aux mois précédents :".format(mois)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": texte}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        improvement_observations = response.choices[0].message.content.strip()
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] les observations d'amélioration ont été générées")
        return improvement_observations

    def generate_report(file_path,nom_mois,old_sum):
         # Lire le fichier CSV avec les données
        df_avis = pd.read_csv(file_path)
        # Convertir la colonne "date" en type de données datetime
        df_avis["date"] = pd.to_datetime(df_avis["date"])

        # convertie un mois comme fevrier en 2
        nb_mois = list(calendar.month_name).index(nom_mois)

        # Filtrer les avis pour l'année 2023
        df_2023 = df_avis[df_avis["date"].dt.year == 2023]
        df_2023 = df_avis[df_avis["date"].dt.month == nb_mois]
        # Extraire les colonnes "nom", "note", "date" et "texte" du DataFrame
        avis = df_2023["texte"].tolist()
        dates = df_2023["date"].tolist()
        notes = df_2023["note"].tolist()
        
        # Convertir les notes en texte
        notes_text = [str(note) for note in notes]

        # Préparer les données d'entrée pour GPT
        input_text = ""
        for i in range(len(avis)):
            input_text += f"{avis[i]} (Date: {dates[i]}) "


        # Génération du résumé, des propositions de points d'amélioration et des observations d'amélioration
        summary = gpt.generate_summary(input_text,nom_mois)
        improvement_points = gpt.generate_improvement_points(input_text,nom_mois)
        improvement_observations = gpt.generate_improvement_observations(summary,old_sum,nom_mois)

        return summary, improvement_points, improvement_observations

    def generate_response(avis):
        # Paramètres de la génération de texte avec le modèle ChatGPT
        model = 'gpt-3.5-turbo'  # Modèle ChatGPT approprié
        max_tokens = 300  # Nombre maximum de tokens dans la réponse générée

        # Définir l'échange conversationnel
        conversation = [
            {'role': 'system', 'content': 'Vous : ' + avis},
            {'role': 'system', 'content': 'Restaurant : '}
        ]

        # Génération de la réponse
        response = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            max_tokens=max_tokens,
            n=1,  # Générer une seule réponse
            stop=None,  # Ne pas arrêter la génération avant la fin
            temperature=0.7,  # Contrôler le niveau de créativité de la réponse (entre 0 et 1)
            frequency_penalty=0.0,  # Pas de pénalité de fréquence pour encourager la diversité des réponses
            presence_penalty=0.0  # Pas de pénalité de présence pour encourager la diversité des réponses
        )

        # Récupération de la réponse générée
        generated_response = response.choices[0].message.content.strip()
        return generated_response

class feeling():
    '''
    Cette classe permet d'analyser le sentiment des avis
    '''
    def analyse_one(avis):
        '''
        Cette fonction analyse le sentiment d'un avis
        '''
        # Initialise le SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Analyse de sentiment pour chaque avis
        score = sia.polarity_scores(avis)
        
        # Décider si l'avis est positif, négatif ou neutre en fonction du score de polarité
        if score['compound'] >= 0.05:
            sentiment = "positif"
        elif score['compound'] <= -0.05:
            sentiment = "négatif"
        else:
            sentiment = "neutre"
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Analyse du sentiment de l'avis : {}".format(sentiment))
        return sentiment

    def nb_sentim(nom_mois,chemin = 'avis/data.csv'):
        '''
        Cette fonction compte le nombre d'avis positifs, négatifs et neutres
        '''
        # recupere le fichier csv avis/data.csv
        df = pd.read_csv(chemin)
        nb_mois = list(calendar.month_name).index(nom_mois)
        df['date'] = pd.to_datetime(df['date'])
        # on filtre les avis du mois passé
        df_mois = df[df['date'].dt.month == nb_mois]
        # on récupère les avis
        review = df_mois['texte'].tolist()
        # on analyse les sentiments
        sentiment_avis = feeling.analyse_all(review)
        # on compte les sentiments
        nb_positif = sentiment_avis.count('positif')
        nb_negatif = sentiment_avis.count('négatif')
        nb_neutre = sentiment_avis.count('neutre')

        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'analyse des sentiments des avis est complète")
        return nb_positif, nb_negatif, nb_neutre

    def analyse_all(avis):
        '''
        Cette fonction analyse le sentiment des avis
        '''
        # Liste des sentiments des avis
        sentiment_avis = []
        # Initialise le SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Analyse de sentiment pour chaque avis
        for avis_texte in avis:
            try :
                score = sia.polarity_scores(avis_texte)
                # Décider si l'avis est positif, négatif ou neutre en fonction du score de polarité
                if score['compound'] >= 0.03:
                    sentiment = "positif"
                elif score['compound'] <= -0.02:
                    sentiment = "négatif"
                else:
                    sentiment = "neutre"
                sentiment_avis.append(sentiment)
            except :
                continue
    
        return sentiment_avis
    
if __name__ == '__main__':
    mois = ['janvier','févrirer','mars','avril','mai']
    rapport.multi_rap(mois)

