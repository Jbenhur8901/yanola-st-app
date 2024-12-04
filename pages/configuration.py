import streamlit as st
import utils.file_processing as fp

st.title("Traitement de fichier")

files = st.file_uploader("Télécharger les fichiers",              
                 type=None, 
                 accept_multiple_files=True
                 )


with st.status("",expanded=True) :
    chunk_size = st.slider("Taille de bloc", min_value=0, max_value=2500)
    chunk_overlap = st.slider("Chevauchement de blocs", min_value=0, max_value=500,help="")
    index_name = st.text_input("entrez le nom de votre index")
    btn = st.button("Traiter")
    
    if btn :
        for file in files :
            if file.type =="application/pdf":
                chunks = fp.load_pdf(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if fp.create_index(index_name) == "Success" : 
                    index = fp.add_documents(index_name, chunks)
                else : 
                    index = fp.add_documents(index_name, chunks)
        st.write(index)
        st.success('This is a success message!', icon="✅")
    
    

