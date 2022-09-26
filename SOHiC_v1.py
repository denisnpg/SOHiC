"""
Created on Mon Nov  2 21:32:18 2020

@author: Lucas
"""

from datetime import date
import streamlit as st
import pandas as pd
import base64
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from pathlib import Path
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from validclust import dunn
import random
from statistics import mean, stdev
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuracion de las imagenes que se descargan en el boton del html
formato = "svg"     # formatos posibles: png, svg, jpeg, webp
altura = 700       # altura de la imagen en pixeles
ancho = 1400        # ancho de la imagen en pixeles
escala_letras = 1   #  Multiply title/legend/axis/canvas sizes by this factor


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='LIDEB Tools - Hierarchical Clustering',
    layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


from PIL import Image
image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
st.subheader(":pushpin:" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")
st.markdown(":computer:""**Web Site** " "<https://lideb.biol.unlp.edu.ar>")


#---------------------------------#
st.write("""
# CHiCA Web App
**CHiCA implements a diversity of classic hierarchical agglomerative clustering approaches. CHiCA allows the user, by use of interactive graphs, an intuitive and visual choice of the number of clusters to consider.**
The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Scikit-learn](https://scikit-learn.org/stable/), [Plotly](https://plotly.com/python/), [Scipy](https://www.scipy.org/)
""")

text = '''
---
'''

st.markdown("""
         **To cite the application, please reference XXXXXXXXX**
         """)

st.markdown(text)


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your datasets SMILES')

uploaded_file_1 = st.sidebar.file_uploader("Upload a TXT file with one SMILES per line", type=["txt"])

# Sidebar - Specify parameter settings
st.sidebar.header('Morgan FP Radio')
fingerprint_radio = st.sidebar.slider('Morgan fingerprint Radio', 2, 4, 2, 1)

st.sidebar.subheader('Morgan FP Lengh')
fingerprint_lenght = st.sidebar.slider('Set the fingerprint lenght', 512, 2048, 1024, 512)

st.sidebar.subheader('Morgan FP with Features')
features = st.sidebar.checkbox('Check if you want the FP to be generated with Features', True, False)
# similarity_metric = st.sidebar.selectbox("Select the similarity metric", ("TanimotoSimilarity", "DiceSimilarity", "CosineSimilarity", "SokalSimilarity", "KulczynskiSimilarity", "McConnaugheySimilarity"),0)

# Método por el cual la matriz de distancia es recalculada una vez que se forma cada cluser. Mas info aca: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
metodo = st.sidebar.selectbox("Clustering method", ("single", "complete", "average", "weighted"),0)

# Metodo para determinar la distancia entre observaciones en un espacio p-dimensional. # Mas info aca: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
# metrica= "euclidean"     # metricas posibles: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

st.sidebar.subheader('Dense clusters')
dense_cluster_limit = st.sidebar.slider('Set % of total compounds to be considered as dense clusters', 2, 50, 5, 1)


# Options for final clustering
ready = st.sidebar.checkbox('Check only if you have already decided the optimal cutoff for clustering')

if ready == True:
    cutoff_clusters = st.sidebar.number_input('Set the cutoff')


#%%

# =============================================================================
# Graphics
# =============================================================================

### Scatter silhouette score ###
def silhouette_plot(df, maximo, limit_dense):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter(x=df["Distance_cutoff"], y=df["Small clusters"],
                        mode='lines+markers',
                        name='Small clusters'))
    fig.add_trace(go.Scatter(x=df["Distance_cutoff"], y=df["Dense clusters"],
                        mode='lines+markers',
                        name='Dense clusters'))
    fig.add_trace(go.Scatter(x=df["Distance_cutoff"], y=df["Outliers"],
                        mode='lines+markers',
                        name='Outliers'))

    fig.add_trace(go.Scatter(x=df["Distance_cutoff"], y=df["Silhouette"].round(2),
                        mode='lines+markers',
                        name='Silhouette'),secondary_y=True)
    
    fig.update_layout(plot_bgcolor = 'rgb(256,256,256)', hovermode='x',
                      legend = dict( font = dict(size=25, family='Calibri', color='black')))
    
    fig.update_xaxes(title_text='Distance cutoff', showgrid=False, 
                      showline=True, linecolor='black', gridcolor='lightgrey',
                      range= [-0.05, 1],
                      tickfont=dict(family='Calibri', size=20, color='black'), ticks = 'outside', tickson = 'labels',
                      title_font = dict(size=28, family='Calibri', color='black'))
    
    fig.update_yaxes(title_text='Number of Molecules', showgrid=False,secondary_y=False,
                      showline=True, linecolor='black', gridcolor='lightgrey', 
                      tickfont=dict(family='Calibri', size=20, color='black'),
                      title_font = dict(size=28, family='Calibri', color='black'))

    fig.update_yaxes(title_text='Silhouette Score', showgrid=False,secondary_y=True,
                      showline=True, linecolor='black', gridcolor='lightgrey', 
                      tickfont=dict(family='Calibri', size=20, color='black'),
                      title_font = dict(size=28, family='Calibri', color='black'))
    st.plotly_chart(fig)
    
    st.markdown("You can download the figures clicking in the camara icon :blush: ")
    st.markdown("**Small clusters** have between 2 and " + str(int(limit_dense)) + " molecules")
    st.markdown("**Dense clusters** have at least " + str(int(limit_dense)) + " molecules")
    st.markdown("We have considered **outliers** to the molecules that do not integrate any clusters")
    return

### Dendrogram plot ###
def dendogram_plot(ddgm, distanceMatrix):
    fig2 = ff.create_dendrogram(ddgm,orientation='bottom',
        linkagefun=lambda x: linkage(distanceMatrix, method=metodo))
    
    fig2.update_layout(width=1600, height=800, plot_bgcolor = 'rgb(256,256,256)')
    fig2.update_xaxes(showgrid=False, 
                      showline=True, linecolor='black', gridcolor='lightgrey',
                      tickfont=dict(family='Calibri', size=16, color='black'), ticks = 'outside', tickson = 'labels',
                      title_font = dict(size=23, family='Calibri', color='black'))
    
    fig2.update_yaxes(title_text='Distance', showgrid=False, 
                      showline=True, linecolor='black', gridcolor='lightgrey', 
                      tickfont=dict(family='Calibri', size=23, color='black'),
                      title_font = dict(size=30, family='Calibri', color='black'))
    
    st.plotly_chart(fig2)
    return

### Bar Plot counting molecules ###
def bar_plot_counts(clusters_final):

    df_bar = clusters_final.copy()
    df_bar['suma'] = [1 for x in range(clusters_final.shape[0])]
    fig3 = px.bar(df_bar, x = 'CLUSTER', y = 'suma')
    
    fig3.update_layout(plot_bgcolor = 'rgb(256,256,256)')
                       
    fig3.update_xaxes(title_text='Cluster Number', showline=True, linecolor='black', 
                      gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                      tickfont=dict(family='Calibri', size=16, color='black'),
                      title_font = dict(size=23, family='Calibri', color='black'))
    
    fig3.update_yaxes(title_text='Amount of molecules', showline=True, linecolor='black', 
                      gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                      tickfont=dict(family='Calibri', size=20, color='black'),
                      title_font = dict(size=30, family='Calibri', color='black'))
    st.plotly_chart(fig3)
    return 

#%%

# =============================================================================
# Clustering
# =============================================================================

# FP calculation
def FP_calculation(df_1):
    list_smiles = df_1[0].tolist()
    lenght_dataset_1 = len(list_smiles)
    st.markdown('Dataset has: ' + str(lenght_dataset_1) + " molecules")
    fps_1 = []

    # Fingerprints
    for m in list_smiles:
        try:
            mol = Chem.MolFromSmiles(m)
            fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol,fingerprint_radio,nBits=fingerprint_lenght,useFeatures=features)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp_1,arr)
            fps_1.append(arr)

        except:
            st.write("We have a problem with molecule: " + str(m))
            st.error('Please review your input molecules or remove the above molecule')
            st.stop()
            
    df_FP = pd.DataFrame(fps_1)
    return df_FP, list_smiles
 

# Dendrogram and silhouette
def dendrogram_and_evaluation(df_FP):
    distanceMatrix = pdist(df_FP.astype(float), metric = 'jaccard')
    ddgm = linkage(distanceMatrix, method=metodo)
    maximo = max(ddgm[:,2])
    n_clusters = list(np.arange(0.0,1,0.05))
    tabla_silhouettes = []
    for x in n_clusters:
        clusters_final = fcluster(ddgm, t=x, criterion='distance')
        unique, counts = np.unique(clusters_final, return_counts=True)
        small_clusters = []
        dense_clusters = []
        limit_dense = round(len(list(clusters_final))*(dense_cluster_limit/100),0)
        outliers_ok= []
        for cluster in list(counts):
            if cluster > 1 and cluster < limit_dense:
                small_clusters.append(1)
            if cluster >= limit_dense:
                dense_clusters.append(1)
            else:
                outliers_ok.append(1)
        try:
            silhouette = silhouette_score(df_FP, clusters_final, metric='jaccard')
            ok = [x,silhouette,len(small_clusters),len(dense_clusters),len(outliers_ok)]
            tabla_silhouettes.append(ok)
        except:
            ok = [x, None ,len(small_clusters),len(dense_clusters),len(outliers_ok)]
            tabla_silhouettes.append(ok)

    tabla_final = pd.DataFrame(tabla_silhouettes)
    tabla_final.rename(columns={0: 'Distance_cutoff', 1:"Silhouette",2:"Small clusters",3:"Dense clusters", 4:"Outliers"},inplace=True)
    
    if ready == False:
        dendogram_plot(ddgm, distanceMatrix)
        silhouette_plot(tabla_final, maximo, limit_dense)
   
    return ddgm

#%%

# =============================================================================
# Clustering performance determination 
# =============================================================================

def clustering_performance(df_FP, clusters_final):
    try:
        silhouette_avg = round(silhouette_score(df_FP, clusters_final,metric='jaccard'),4)
        db_score = round(davies_bouldin_score(df_FP, np.ravel(clusters_final)),4)
        ch_score = round(calinski_harabasz_score(df_FP, np.ravel(clusters_final)),4)
        dist_dunn = cdist(df_FP,df_FP, metric= 'jaccard')
        dunn_score = round(dunn(dist_dunn, np.ravel(clusters_final)),4)
    except:
        pass

    # only for random clusters
    validation_round = [silhouette_avg, db_score, ch_score, dunn_score]
    sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st = cluster_random(df_FP, clusters_final, dist_dunn)
    random_means = [sil_random,db_random,ch_random,dunn_random]
    random_sds = [sil_random_st,db_random_st,ch_random_st,dunn_random_st]
    
    table_metrics = pd.DataFrame([validation_round,random_means,random_sds]).T
    table_metrics=table_metrics.rename(index={0: 'Silhouette score', 1:"Davies Bouldin score", 2: 'Calinski Harabasz score', 3:'Dunn Index'},columns={0:"Value",1:"Mean Random",2:"SD Random"})

    st.write(table_metrics)
    st.markdown(filedownload3(table_metrics), unsafe_allow_html=True)
    st.markdown("-------------------")
    
    return validation_round


def cluster_random(df_FP, clusters_final, dist_dunn):
    compilado_silhoutte = []
    compilado_db = []
    compilado_ch = []
    compilado_dunn = []
    
    for i in range(100):
        random.seed(a=i, version=2)
        random_clusters = []
        for x in clusters_final:
            random_clusters.append(random.randint(0,len(set(clusters_final))-1))
        try:
            silhouette_random = silhouette_score(df_FP, np.ravel(random_clusters))
            compilado_silhoutte.append(silhouette_random)
            db_random = davies_bouldin_score(df_FP, np.ravel(random_clusters))
            compilado_db.append(db_random)
            ch_random = calinski_harabasz_score(df_FP, np.ravel(random_clusters))
            compilado_ch.append(ch_random)
            dunn_random = dunn(dist_dunn, np.ravel(random_clusters))
            compilado_dunn.append(dunn_random)

        except:
            pass

    sil_random = round(mean(compilado_silhoutte),4)
    sil_random_st = str(round(stdev(compilado_silhoutte),4))
    db_random = round(mean(compilado_db),4)
    db_random_st = str(round(stdev(compilado_db),4))
    ch_random = round(mean(compilado_ch),4)
    ch_random_st = str(round(stdev(compilado_ch),4))
    dunn_random = round(mean(compilado_dunn),4)
    dunn_random_st = str(round(stdev(compilado_dunn),4))

    return sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st

#%% File downloads

def filedownload(df_FP, ddgm, list_smiles):
    clusters_final = fcluster(ddgm, t=cutoff_clusters, criterion='distance')
    clusters_final1 = pd.DataFrame(clusters_final)
    clustering_performance(df_FP, clusters_final)
    clusters_final = pd.concat((pd.DataFrame(clusters_final1.index.values.tolist(), columns=['Molecule']),pd.DataFrame(clusters_final, columns=['CLUSTER'])), axis=1)
    clusters_final.index = list_smiles
    csv = clusters_final.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    st.markdown(":point_down: **Here you can dowload the cluster assignation**", unsafe_allow_html=True)
    href = f'<a href="data:file/csv;base64,{b64}" download="cluster_assignation.csv">Download CSV File with your clusters</a>'
    return href

def filedownload2(df):
    csv = df.to_csv(index=False,header=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering_settings.csv">Download CSV File with your clustering settings</a>'
    return href

def filedownload3(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Validations.csv">Download CSV File with the validation metrics</a>'
    return href




#%%
### Settings file ###

def setting_info():
    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date clustering was performed: " , fecha])
    settings.append(["Seetings:",""])
    settings.append(["Morgan FP Radio:", str(fingerprint_radio)])
    settings.append(["Morgan FP Lengh:", str(fingerprint_lenght)])
    settings.append(["Morgan FP with Features:", str(features)])
    # settings.append(["Similarity metric:", str(similarity_metric)])
    settings.append(["Clustering method:", str(metodo)])
    settings.append(["% of total compounds to be considered as dense clusters:", str(dense_cluster_limit)])
    settings.append(["Cutoff clusters:", str(cutoff_clusters)])
    settings.append(["",""])
    settings.append(["To cite the application, please reference: ","XXXXXXXXXXX"])   
    settings_df = pd.DataFrame(settings)
    return settings_df

#%%
# ---------------------------------#

if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        df_1 = pd.read_csv(uploaded_file_1,sep="\t",header=None)
        df_FP, list_smiles = FP_calculation(df_1)
        ddgm = dendrogram_and_evaluation(df_FP)
        if ready == False:
            st.markdown('**Once you have identified the optimal cutoff, re-run the clustering but checking the option of "optimal cutoff" ** :exclamation: :exclamation: :exclamation:')
        if ready == True:          
            if cutoff_clusters is not None:
                st.markdown(filedownload(df_FP,ddgm,list_smiles), unsafe_allow_html=True)
                st.markdown("-------------------")
                settings_df = setting_info()
                st.markdown(":point_down: **Here you can download your settings**", unsafe_allow_html=True)
                st.markdown(filedownload2(settings_df), unsafe_allow_html=True)

# Example file
else:
    st.info('Awaiting for TXT file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        df_1 = pd.read_csv("molecules_1.txt",sep="\t",header=None)
        df_FP, list_smiles = FP_calculation(df_1)
        ddgm = dendrogram_and_evaluation(df_FP)
        if ready == False:
            st.markdown('**Once you have identified the optimal cutoff, re-run the clustering but checking the option of "optimal cutoff" ** :exclamation: :exclamation: :exclamation:')
        if ready == True:          
            if cutoff_clusters is not None:
                st.markdown(filedownload(df_FP,ddgm,list_smiles), unsafe_allow_html=True)
                st.markdown("-------------------")
                settings_df = setting_info()
                st.markdown(":point_down: **Here you can download your settings**", unsafe_allow_html=True)
                st.markdown(filedownload2(settings_df), unsafe_allow_html=True)


#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; ' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display:; text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

