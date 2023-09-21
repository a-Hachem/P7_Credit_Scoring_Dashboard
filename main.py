import streamlit as st
import requests
# import altair as alt
import pandas as pd
import numpy as np
# import time
import matplotlib.pyplot as plt
from joblib import load
import pickle
import base64
from streamlit_shap import st_shap
import shap
import plotly.graph_objects as go

check = 0
global lien 
# lien = "http://127.0.0.1:5000/" 
lien = "https://projet7-ml-webapi.onrender.com/"

def get_predictions_from_api(id):
    
    api_url_1 = f"{lien}predict?client_id={id}"
    api_url_2 = f"{lien}predict_prob?client_id={id}"
    
    response_1 = requests.post(api_url_1)
    response_2 = requests.post(api_url_2)
    
    if response_1.status_code == 200:
        prediction = response_1.json()["prediction"]
        prediction_prob = float(response_2.json()["prediction_prob"])
        prediction_prob = round(prediction_prob, 6)
        return prediction, prediction_prob
    else:
        st.error(f"Erreur lors de la requête à l'API. Code de statut : {response_1.status_code}")
        st.error(f"Contenu de la réponse : {response_1.text}")
        return None
    
def similar_client_from_api(id):
    api_url = f"{lien}similar_cl?client_id={id}"    
    response = requests.post(api_url)        
    if response.status_code == 200:
        json_data = response.json()
        df = pd.DataFrame.from_records(json_data)
        return df
    else:
        st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")
        st.error(f"Contenu de la réponse : {response.text}")
        return None

def shap_values_from_api(id):
    api_url = f"{lien}shap_val?client_id={id}"    
    response = requests.post(api_url)        
    if response.status_code == 200:
        shap_values_encoded = response.json()["shape_values"]
        shap_values_bytes = base64.b64decode(shap_values_encoded)
        shap_values = pickle.loads(shap_values_bytes)
        return shap_values
    else:
        st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")
        st.error(f"Contenu de la réponse : {response.text}")
        return None

def shap_values_all_from_api(id):
    api_url = f"{lien}shap_val_all"    
    response = requests.post(api_url)        
    if response.status_code == 200:
        shap_values_all_encoded = response.json()["shape_values_all"]
        shap_values_all_bytes = base64.b64decode(shap_values_all_encoded)
        shap_values_all = pickle.loads(shap_values_all_bytes)
        return shap_values_all
    else:
        st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")
        st.error(f"Contenu de la réponse : {response.text}")
        return None

               
with st.sidebar:
    st.title("Benvenue au Dashboard !")
    id = st.text_input("Entrez l'ID du Client")
    st.text("Que voulez vous visualiser ?")
    check1 = st.checkbox('Décision prise', value = True)
    check2 = st.checkbox('Probabilités associées', value = True)
    check3 = st.checkbox('5 plus proches clients', value = False)
    check4 = st.checkbox('Valeurs SHAP', value = False)
    check5 = st.checkbox('Valeurs SHAP globales', value = False)
    if st.button('Envoyer'):
        check = 1


if check == 1:
    with st.container():
            if int(id) > 0 :

                result, proba = get_predictions_from_api(int(id))

                if check1:
                    st.markdown('### Décision prise')
                    if int(result)==1:
                        st.success('Votre demande a été acceptée', icon="✅")
                    elif int(result)==0:
                        # st.warning('Votre demande a été rejetée', icon="⚠️")
                        st.error('Votre demande a été rejetée', icon="⚠️")
                
                if check2:
                    
                        st.markdown("### Probablilités associées")
                        categories = ["Rejeté", "Accepté"]
                        probabilities = [1-proba, proba]
                        pourcentages = ['{:.2%}'.format(1-proba), '{:.2%}'.format(proba)]
                    
                        colors = ['#ff9999','#66b3ff'] 

                        # ========================== pyplot =========
                        fig = go.Figure(data=[go.Pie(labels=categories, values=probabilities, hole=.3, marker=dict(colors=colors))])

                        # Afficher le graphique avec la largeur du conteneur Streamlit
                        st.plotly_chart(fig, use_container_width=True)

                        
                        # ============================ matplotlib =====
                        # fig1, ax1 = plt.subplots()                  # Créer un graphique en forme de donut
                        # ax1.pie(probabilities, colors = colors, labels=categories, autopct='%1.1f%%', startangle=90)
                        # centre_circle = plt.Circle((0,0),0.70,fc='white')
                        # fig = plt.gcf()
                        # fig.gca().add_artist(centre_circle)

                        # ax1.axis('equal')                           # Diviser le graphique en deux parties
                        # plt.tight_layout()

                        # st.pyplot(fig1, use_container_width=True)   # Afficher le graphique avec la largeur du conteneur
                        
                        # ============================ altair =========
                        # source = pd.DataFrame({"category": categories, "value": probabilities, "label": pourcentages})

                        # pie = alt.Chart(source).mark_arc(innerRadius=75).encode(
                            # theta=alt.Theta(field="value", type="quantitative", stack=True, scale=alt.Scale(type="linear",rangeMax=1.5708, rangeMin=-1.5708 )),
                            # color=alt.Color(field="category", type="nominal"),
                        # ).properties(
                            # height=200, width=400,
                            # title="Probabilités de Prédiction"
                            # width=alt.Step(40)
                        # )

                        # pie = pie + pie.mark_text(radius=175, fontSize=16).encode(text='label')
                        # st.altair_chart(pie, use_container_width=True)
                
                 
                # Récupérer les 5 plus proches clients (à partir de l'api)
                if check3:
                    st.markdown("### 5 plus proches clients")
                    st.write(similar_client_from_api(int(id)))
                
                if check4:
                    st.markdown("### Valeurs SHAP")
                    shap_values = shap_values_from_api(int(id)) 
                    # st_shap(shap.summary_plot(shap_values))
                    plt.subplot()
                    st_shap(shap.plots.waterfall(shap_values[0]))

                if check5:
                    st.markdown("### Valeurs SHAP globales")
                    shap_values_all = shap_values_all_from_api(int(id)) 
                    plt.subplot()
                    st_shap(shap.summary_plot(shap_values_all))
                
                

else:
    print("")
        