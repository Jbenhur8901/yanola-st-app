import streamlit as st
import utils.yanola as yl

# === Initialisation des variables dans la session === #
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Bonjour, je suis Yanola, comment puis-je vous aider ?",
                "avatar": "images/Hi im YANOLA BCA.png",
            }
        ]
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = yl.generate_id()

# === Barre latérale avec options === #
def render_sidebar():
    with st.sidebar:
        st.logo("images/logo nodes (1).png", size="large")  # Optimisation visuelle
        st.write("# Options")
        st.markdown("---")
        
        index = "nsiadirect" #st.text_input("Entrez le nom de l'index", key="index_input")
        subject = "Les offres et services de NSIA" #st.text_input("Entrez le thème principal", key="subject_input")
        with open("prompt.txt","r") as prompt :
            sys_instructions = prompt.read() #st.text_area("Entrez vos instructions", key="sys_instructions_input")
        
        return index, subject, sys_instructions

# === Afficher les messages de chat === #
def render_chat_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])

# === Ajouter un message au chat === #
def add_chat_message(role, content, avatar, stream = None):
    st.session_state.messages.append({"role": role, "content": content, "avatar": avatar})
    if stream:
        st.chat_message(role, avatar=avatar).write_stream(stream)
    else:
        st.chat_message(role, avatar=avatar).write(content)

# === Fonction principale du chatbot === #
def handle_chat_input(prompt, index, subject, sys_instructions):
    try:
        # Initialiser l'agent Yanola
        retrieval_agent = yl.RetrievalAgent(
            user_input = prompt, 
            index_name = index, 
            sys_instruction=sys_instructions, 
            chat_id=st.session_state["chat_id"], 
            subject_matter=subject
        )
        # Lancer l'agent
        response = retrieval_agent.run_agent()
        stream = retrieval_agent.stream_data
        
        # Ajouter la réponse dans le chat
        add_chat_message("assistant", response, "images/Hi im YANOLA BCA.png", stream=stream)
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de l'agent : {e}")

# === Application principale === #
def main():
    # Initialisation
    initialize_session_state()

    # Afficher la barre latérale et récupérer les paramètres
    index, subject, sys_instructions = render_sidebar()

    # Afficher le logo au début (uniquement au premier rendu)
    if "logo_shown" not in st.session_state:
        st.image("images/logo nodes (1).png", width=200)
        st.session_state["logo_shown"] = True

    # Afficher les messages
    render_chat_messages()

    # Entrée utilisateur pour le chat
    prompt = st.chat_input(placeholder="Message à Yanola")
    if prompt:
        # Ajouter le message utilisateur dans le chat
        add_chat_message("user", prompt, "images/5049207_avatar_people_person_profile_user_icon (1).png")
        
        # Gérer la réponse
        if not index or not subject or not sys_instructions:
            st.warning("Veuillez remplir tous les champs dans la barre latérale avant de continuer.")
        else:
            handle_chat_input(prompt, index, subject, sys_instructions)

# Exécuter l'application principale
if __name__ == "__main__":
    main()


