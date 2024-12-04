#Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
#Prompts
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
#Agents
from langchain.agents import AgentExecutor, create_tool_calling_agent
#Vectordb
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
#Memory
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
# Document Compression
from langchain.retrievers.document_compressors import LLMChainExtractor
# Prompt
from langchain.prompts import PromptTemplate
#Custom tools
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
#Query Transformation 
from langchain.retrievers.multi_query import MultiQueryRetriever
#Environnement Variable
import logging
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

class RetrievalAgent:
    """
    A retrieval agent designed to extract relevant documents based on user queries.

    This class manages the retrieval process, including fetching chat history, setting up retrieval tools, and executing the agent.

    Parameters:
    - user_input (str): The user query.
    - index_name (str): The name of the vector index.
    - sys_instruction (str): System instruction message.
    - chat_id (str): Unique identifier for the chat session.
    - subject_matter (str): Subject matter of the queries.
    - model (str): Name of the language model to be used for chat responses (default is "gpt-3.5-turbo-0125").
    - timeline (int): Time limit for chat history retrieval (default is 300 seconds).
    - voyageai_model (str): Name of the VoyageAI model for embeddings (default is "voyage-large-2").
    - temperature (float): Temperature parameter for language model (default is 0.5).

    Methods:
    - get_chat_history(): Retrieve the chat history from the storage.
    - nodes_pipeline(question: str) -> list: Pipeline for extracting relevant documents based on the user's question.
    - set_retrieval_tool(): Set up the retrieval tool.
    - set_tools(): Set up additional tools.
    - run_agent(): Run the retrieval agent to process user queries and return relevant documents.
    """

    def __init__(self, user_input: str, index_name: str, sys_instruction: str, chat_id: str, subject_matter: str, model="gpt-4o-mini",timeline: int = 300, voyageai_model: str = "voyage-large-2-instruct", temperature: float = 0.2):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:

            # conditions to define the llm model based on type
            self.model = model
            self.temperature = temperature
            if self.model.startswith("gpt"):
                self.llm = ChatOpenAI(model=self.model,temperature=self.temperature, api_key=st.secrets["api_keys"]["openai_api_key"])
            elif self.model.startswith("claude"):
                self.llm = ChatAnthropic(model_name=model,temperature=self.temperature, api_key=st.secrets["api_keys"]["anthropic_api_key"])
            else:
                self.llm = ChatOpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ["TOGETHER_API_KEY"], model=self.model)

            self.embedding = VoyageAIEmbeddings(model=voyageai_model, api_key=st.secrets["api_keys"]["voyageai_api_key"])
            self.user_input = user_input
            self.index_name = index_name
            self.sys_instruction = sys_instruction
            self.subject_matter = subject_matter
            self.chat_id = chat_id
            self.timeline = timeline
            self.get_chat_history()
            self.set_retrieval_tool()
            self.set_tools()

        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")

    def get_chat_history(self):

        try:

            url = st.secrets["api_keys"]["upstash_url"]
            token = st.secrets["api_keys"]["upstash_token"]
            self.upstash_client = UpstashRedisChatMessageHistory(url=url, token=token, ttl=self.timeline, session_id=self.chat_id)
            self.chat_history = self.upstash_client.messages

        except Exception as e:
            self.logger.error(f"Error fetching chat history: {e}")
            self.chat_history = []

    def nodes_pipeline(self, question) -> list:

        try:
            # Define prompts for query transformation and context compression
            prompt_query = """Tu es un assistant de modèle linguistique d'IA. Ta tâche consiste à générer quatre versions différentes de la question posée par l'utilisateur afin d'extraire les documents pertinents 
            d'une base de données vectorielle. En générant des perspectives multiples sur la question de l'utilisateur,ton objectif est d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité
            basée sur la distance. Enumère les directement un à un sans commentaire sans créer de nouvelles lignes.
        
            question originale: {question}
            """
            prompt_compressor = """Considérant la question et le contexte suivants, extraire toute partie du contexte *EN L'ÉTAT* qui est pertinente pour répondre à la question. Si aucune partie du contexte n'est pertinente,
            renvoie NO_OUTPUT. \n\n - Tu ne dois jamais modifier les parties extraites du contexte.\n- Question : {question}\n- Contexte:\n>>{context}\n>>\nParties pertinentes extraites :"""
            query_transformation_prompt = PromptTemplate(template=prompt_query, input_variables=["question"])
            context_compression_prompt = PromptTemplate(template=prompt_compressor, input_variables=["question", "context"])

            # Set up vector store and retriever
            vector_store = PineconeVectorStore(embedding=self.embedding, index_name=self.index_name, pinecone_api_key=st.secrets["api_keys"]["pinecone_api_key"])
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Initialize language model for query transformation
            llm = ChatOpenAI(model="gpt-4o-mini",temperature=self.temperature, api_key=st.secrets["api_keys"]["openai_api_key"]) #ChatAnthropic(model_name="claude-3-haiku-20240307",temperature=self.temperature)
            #llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620",temperature=self.temperature)
            

            # Transform query and compress context
            retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, prompt=query_transformation_prompt)
            unique_docs = retriever_from_llm.invoke(question)
            compressor = LLMChainExtractor.from_llm(llm=llm, prompt=context_compression_prompt)
            self.compressed_docs = compressor.compress_documents(documents=unique_docs, query=question)
            return self.compressed_docs
        
        except Exception as e:
            # Log error and return empty list
            self.logger.error(f"Error in nodes_pipeline: {e}")
        return []


    def set_retrieval_tool(self):

        try:
            class PipelineInputs(BaseModel):
                question : str = Field(description=f"La question posée sur {self.subject_matter}")

            self.nodes_pipe = StructuredTool.from_function(func=self.nodes_pipeline,
                                                  name="nodes_pipeline",
                                                  description="""Cet outil utilise des techniques de recherche avancées pour récupérer les documents pertinents 
                                                  à partir d'une base de données vectorielles en fonction des questions posées. Il est indispensable d'utiliser 
                                                  cet outil pour obtenir le contexte pertinent lorsque l'on répond à des questions rélatives à {self.subject_matter}.""",
                                                  args_schema=PipelineInputs
                                                  )
        except Exception as e:
            self.logger.error(f"Error setting up retrieval tool: {e}")

    def set_tools(self):
        """
        Set up additional tools.
        """
        try:
            self.tools = [self.nodes_pipe]
        except Exception as e:
            self.logger.error(f"Error setting up additional tools: {e}")

    def run_agent(self):
        import re
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                ("system", f"{self.sys_instruction}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "<input>{input}</input>"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
                ]
            )

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools)
            query = executor.invoke({"input": self.user_input, "chat_history": self.chat_history})
            self.upstash_client.add_messages([HumanMessage(content=self.user_input), AIMessage(content=query["output"])])
            self.clean_text = re.sub(r'<[^>]+>\n?', '', query["output"])
            return self.clean_text
        except Exception as e:
            self.logger.error(f"Error running agent: {e}")
            
    def stream_data(self):
        import time
        for word in self.clean_text.split(" "):
            yield word + " "
            time.sleep(0.02)
            
def generate_id():
    import re
    import requests as rq  
    
    uuid = rq.get("https://www.uuidtools.com/api/generate/v4")
    data = str(uuid.content)
    cleaned_data = re.sub(r'b\'\[\"|\"\]\'', '', str(data))
    return cleaned_data

  
