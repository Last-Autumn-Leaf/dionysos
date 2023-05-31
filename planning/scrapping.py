'''
Ce code permet de récupérer les données du planning de Planifico.
Les données sont enregistrées dans un fichier CSV.

Pour exécuter ce code, il faut installer les librairies suivantes :
* selenium
* bs4
* requests

Auteur : Ilyas Baktache
'''

# import des librairies


# Enregistrement des données
import csv
import re
# Pour gérer les dates et le temps
from datetime import datetime, timedelta
import time
import locale
# Définir la langue française
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Scrapping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# log
import logging as lg
lg.basicConfig(level=lg.INFO)

# Pour gérer les exceptions
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException

# system
import sys

'''
Class de scrapping
'''

# Les fonction ci-dessous sont des fonctions de scrapping qui permettent de récupérer les données du planning de Planifico.
# Ces fonctions sont appelées par la fonction principale "scrap_planning".

class planifico:
    def planning_data2csv(datas):

        '''
        Cette fonction permet de convertir les données du planning en un fichier CSV.

        Paramètres
        ----------
        datas : dict
            Un dictionnaire contenant les données de planning.

        Retour
        -------
        None
        '''

        # Nom du fichier CSV de sortie
        nom_fichier_csv = 'planning/data_planning.csv'

        # Liste des en-têtes de colonnes
        entetes = ['date', 'roles']

        # Écrire les données dans le fichier CSV
        with open(nom_fichier_csv, mode='w', newline='') as fichier_csv:
            writer = csv.writer(fichier_csv)
            
            # Écrire les en-têtes de colonnes
            writer.writerow(entetes)
            
            # Parcourir les données de chaque jour
            for date, data in datas.items():
                for heure, roles in data.items():
                    # Convertir la date et l'heure en une seule chaîne de caractères
                    date_heure = date.strftime("%Y-%m-%d") + " " + heure.strftime("%H:%M")
                    
                    # Écrire la ligne dans le fichier CSV
                    row = [date_heure, ', '.join(roles)]
                    writer.writerow(row)
        print("Le fichier CSV a été enregistré avec succès.")

    def process_element(driver,element,data,role_count,jour,mois,annee,time_refresh,charge_time):
        """
        Cette fonction permet de traiter les données d'un élément de calendrier.
        
        Paramètres
        ----------
        * driver : WebDriver
            Le navigateur Web.
        * element : WebElement
            L'élément de calendrier à traiter.
        * data : dict
            Un dictionnaire contenant les données de calendrier.
        * role_count : dict
            Un dictionnaire contenant le nombre de fois que chaque rôle de travail a été vu.
        * jour : int
            Le jour du mois.
        * mois : int
            Le mois.
        * annee : int
            L'année.
        * time_refresh : int
            Le délai d'attente pour que la page se charge complètement.
        * charge_time : int
            Le délai d'attente pour que les données de calendrier se chargent.


        Retour
        -------
        * data : dict
            Un dictionnaire contenant les données de calendrier.
        * role_count : dict
            Un dictionnaire contenant le nombre de fois que chaque rôle de travail a été vu.
        """
        numbered_roles = []
        # Attendre que la page se charge complètement (ajuster le délai si nécessaire)
        driver.implicitly_wait(time_refresh)
        # Extraire les données de temps de début et de fin
        
        # Recherche de l'élément "start_time" en utilisant différentes méthodes de localisation
        try:
            start_time_element = element.find_element(By.CSS_SELECTOR, 'span.font-lighter')
        except NoSuchElementException:
            print("erreur11")
            try:
                start_time_element = element.find_element(By.XPATH, './span[@class="font-lighter"]')
            except NoSuchElementException:
                try:
                    start_time_element = element.find_element(By.CLASS_NAME, 'font-lighter')
                except NoSuchElementException:
                    print("erreur")
                    # Si aucun des sélecteurs ne fonctionne, retourner les données vides
                    return data, role_count
        
        start_time = start_time_element.text
        # Recherche de l'élément "end_time" en utilisant différentes méthodes de localisation
        try:
            end_time_element = element.find_element(By.CSS_SELECTOR, 'span.font-lighter.quick-calendarUI3-r0001')
        except NoSuchElementException:
            print("erreur1")
            try:
                end_time_element = element.find_element(By.XPATH, './span[@class="font-lighter quick-calendarUI3-r0001"]')
            except NoSuchElementException:
                try:
                    end_time_element = element.find_element(By.CLASS_NAME, 'quick-calendarUI3-r0001')
                except NoSuchElementException:
                    print("erreur")
                    # Si aucun des sélecteurs ne fonctionne, retourner les données vides
                    return data, role_count
        
        end_time = end_time_element.text
        # Extraire l'heure de début en utilisant une expression régulière
        match_start = re.search(r'(\d{1,2})h(\d{2})', start_time)
        if match_start:
            start_hour = int(match_start.group(1))
            start_minute = int(match_start.group(2))
            start_datetime = datetime(year=annee, month=mois, day=jour, hour=start_hour, minute=start_minute)
            if start_hour < 10:
                start_datetime += timedelta(days=1)
        else:
            pass
        # Extraire l'heure de fin en utilisant une expression régulière
        match_end = re.search(r'(\d{1,2})h(\d{2})', end_time)
        if match_end:
            end_hour = int(match_end.group(1))
            end_minute = int(match_end.group(2))
            if end_hour < 10:
                end_datetime = datetime(year=annee, month=mois, day=jour, hour=end_hour, minute=end_minute) + timedelta(days=1) 
                if 0<=end_hour<4:
                    end_datetime += timedelta(minutes=30)
            else : 
                end_datetime = datetime(year=annee, month=mois, day=jour, hour=end_hour, minute=end_minute)
        else:
            pass
        # Extraire les rôles de travail
        role_elements = element.find_elements_by_css_selector('div.quick-calendarUI2-table1-element1-title2 span')
        roles = [role_element.text.strip() for role_element in role_elements if role_element.text.strip()]
        # Imprimer les rôles par heure
        print(f"Jour: {jour}/{mois}/{annee}")
        print(f"Heure de début: {start_time}")
        print(f"Heure de fin: {end_time}")
        print(f"Rôles: {roles}")
        print()

        for role in roles:
            if role in role_count:
                role_count[role] += 1
                numbered_role = f"{role}_{role_count[role]}"
            else:
                role_count[role] = 1
                numbered_role = role
            numbered_roles.append(numbered_role)
        # Parcourir toutes les heures avec un intervalle de 30 minutes entre l'heure de début et l'heure de fin
        current_datetime = start_datetime
        while current_datetime < end_datetime:
            # Ajouter les rôles de travail à l'heure correspondante dans le dictionnaire
            if current_datetime not in data:
                data[current_datetime] = []

            data[current_datetime].extend(numbered_roles)

            # Incrémenter de 30 minutes
            current_datetime += timedelta(minutes=30)
        
        return data, role_count
        
    def process_day(driver,jour_element,jours_semaine_element,datas,time_refresh,charge_time):
        """
        Traiter les données de calendrier pour un jour donné.

        Paramètres
        ----------
        * driver : selenium.webdriver.chrome.webdriver.WebDriver
            Le navigateur Chrome.
        * jour_element : selenium.webdriver.remote.webelement.WebElement
            L'élément du jour à traiter.
        * jours_semaine_element : selenium.webdriver.remote.webelement.WebElement
            L'élément de la semaine à traiter.
        * datas : dict
            Les données de calendrier.
        * time_refresh : int
            Le temps d'attente pour le chargement de la page.
        * charge_time : int
            Le temps d'attente pour le chargement de la page.

        Retourne
        -------
        * datas : dict
            Les données de calendrier.
        """    
        # On attends 2 secondes pour que la page se charge 
        time.sleep(charge_time)
        # Attendre que la page se charge complètement (ajuster le délai si nécessaire)
        driver.implicitly_wait(time_refresh)
        # Cliquer sur le jour
        jour_elements = jours_semaine_element.find_elements_by_css_selector('div.itemGroup1-item.weekbar1-item.selectable1')
        ActionChains(driver).move_to_element(jour_element).click().perform()
        # Attendre que la date affichée soit visible
        date_element = WebDriverWait(driver, time_refresh).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'span.font-light.padding-y-medium'))
        )
        try :
            # Convertir la date en objet datetime en utilisant le format spécifique
            date_object = datetime.strptime(date_element.text, "%a, %d %b %Y")
        except:
            # Convertir la date en objet datetime en utilisant le format spécifique
            date_object = datetime.strptime(date_element.text, "%a, %d %B %Y")
        
        # Extraire le jour, le mois et l'année de l'objet datetime
        jour = date_object.day
        mois = date_object.month
        annee = date_object.year
        
        # Attendre que la page se charge complètement (ajuster le délai si nécessaire)
        driver.implicitly_wait(time_refresh)
        # Trouver tous les éléments div ayant la classe "colTableCol"
        elements = driver.find_elements_by_css_selector('div.colTableCol')
        
        # Créer un dictionnaire pour stocker les informations sur les rôles de travail pour chaque heure
        data = {}
        role_count = {}
        # Parcourir les éléments et extraire les données de temps et les rôles de travail
        for element in elements:
            data, role_count = planifico.process_element(driver,element,data,role_count,jour,mois,annee,time_refresh,charge_time)
        # Supprimer les doublons de rôles pour chaque heure
        for times, roles in data.items():
            data[times] = list(set(roles))
        
        # Ajouter les données du jour au dictionnaire datas
        datas[date_object] = data
        return datas

    def get_data(password,path_driver =r'/Users/mac2021/Documents/Projet/Projet Dionysos/dionysos/planning/chromedriver', username = "baktacheilyas@gmail.com",oldest_date = datetime(2023, 5, 1),time_refresh = 50,charge_time = 1):
        '''
        Cette fonction permet de récupérer les données de planifico

        Parameters
        ----------
        * path_driver : str
            Chemin vers le driver de chrome
        * username : str
            Nom d'utilisateur
        * password : str
            Mot de passe
        * oldest_date : datetime
            Date la plus ancienne à récupérer
        * time_refresh : int
            Temps d'attente pour le chargement de la page
        * charge_time (forcé) : int
            Temps d'attente pour le chargement de la page

        Returns
        -------
        * datas : dict
            Dictionnaire contenant les données de planifico
        '''
        
        # Initialiser le navigateur Chrome
        driver = webdriver.Chrome(path_driver)

        # Accéder à la page de connexion
        driver.get('https://app.planifico.co/#/')

        '''
        Connection à l'application
        '''

        lg.info("Page de connexion chargée")
        driver.implicitly_wait(time_refresh)
        # Attendre que le formulaire de connexion soit chargé
        WebDriverWait(driver, time_refresh).until(EC.presence_of_element_located((By.ID, "frontpage-login-loginform")))

        # Remplir le formulaire de connexion avec votre email et mot de passe
        email_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Courriel']")
        email_input.send_keys(username)

        password_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Mot de passe']")
        password_input.send_keys(password)

        '''
        Choisir l'entreprise
        '''
        # Attendre que le formulaire de connexion soit chargé
        WebDriverWait(driver, time_refresh).until(EC.presence_of_element_located((By.ID, "frontpage-login-loginform-field-button")))

        # Soumettre le formulaire de connexion
        login_button = driver.find_element(By.ID, "frontpage-login-loginform-field-button")
        login_button.click()

        '''
        Choisir l'onglet horaire
        '''
        # Attendre que le formulaire de connexion soit chargé
        WebDriverWait(driver, time_refresh).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class='company-item-selector company-item-selector-button1 button-o']")))

        button = driver.find_element(By.CSS_SELECTOR, "div[class='company-item-selector company-item-selector-button1 button-o']")
        button.click()

        '''
        Choisir l'onglet Eq.jour
        '''
        # Attendre que le formulaire de connexion soit chargé
        WebDriverWait(driver, time_refresh).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class='company-item-selector company-item-selector-button1 button-o']")))

        button = driver.find_element_by_xpath("//div[@class='formBottomNavigator1-controls-item-label font-small'][contains(text(),'Horaire')]")
        button.click()

        '''
        Choisir l'onglet Eq.jour
        '''
        # Attendre que la page se charge complètement (ajuster le délai si nécessaire)
        driver.implicitly_wait(time_refresh)
        bouton = driver.find_element_by_xpath("//div[@class='formDbHeader1-item mas-active-toggle-black-a10  formDbHeader1-item-2']")
        bouton.click()

        # Attendre que la page se charge complètement (ajuster le délai si nécessaire)
        driver.implicitly_wait(time_refresh)

        '''
        Lancement du scrapping
        '''
        # Attendre que l'élément contenant les jours de la semaine soit visible
        jours_semaine_element = WebDriverWait(driver, time_refresh).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.div-wrapper.bgc-theme-main-darker'))
        )

        # Trouver tous les éléments des jours de la semaine
        driver.implicitly_wait(time_refresh)
        jour_elements = jours_semaine_element.find_elements_by_css_selector('div.itemGroup1-item.weekbar1-item.selectable1')

        # On initialise la date à la date du jour (celle affichée sur la page)
        date_object = datetime.now()

        # Grand dictionnaire pour stocker les données de tous les jours
        datas = {}

        #  On définis une boucle while jusqu'a la date la plus vielle dont on veut récuperer les données
        while date_object > oldest_date: 
            # Parcourir les éléments des jours de la semaine et cliquer sur chaque jour
            for jour_element in jour_elements:
                datas = planifico.process_day(driver,jour_element,jours_semaine_element,datas,time_refresh,charge_time)

            # On enregistre les données à chaque semaine
            planifico.planning_data2csv(datas)

            # On appuie sur le bouton pour revenir au semaines précédentes
            driver.implicitly_wait(time_refresh)
            bouton_element = driver.find_element_by_css_selector('div.position-absolute.position-top-left.height-100.align-center.mas-active-toggle-black-a10')
            bouton_element.click()

            # On attends 2 secondes pour que la page se charge 
            time.sleep(charge_time)

        # On enregistre les données dans un fichier csv
        planifico.planning_data2csv(datas)

        # Fermer le navigateur
        driver.quit()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        password = str(sys.argv[1]) 
    
    planifico.get_data(password)