# Importaciones estándar
import sys
import os
import shutil
import uuid
import tempfile
import logging
from typing import Any, Dict

# Importaciones de terceros
import streamlit as st
from decouple import config
import together
from PyPDF2 import PdfReader
from pydantic import BaseModel, root_validator

# Importaciones para la implementación con Langchain
# Nota: Asegúrese de tener una versión compatible de pydantic (<2)
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Corrección específica para despliegue en Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Configuración de la página en Streamlit
st.set_page_config(page_title='Pregunta a tu PDF', 
                   page_icon="📄", 
                   layout="wide")

# Aplicar CSS para el fondo de la aplicación Streamlit
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("");
            background-attachment: fixed;
            background-size: cover
        }}
    </style>
    """, 
    unsafe_allow_html=True
)

# Título de la página en Streamlit
st.title('🔗 Pregunta a tu PDF 📄')

# Configuración para Together API
together.api_key = config("TOGETHER_API_KEY")

# Definición de la clase TogetherLLM, una subclase de LLM
class TogetherLLM(LLM):
    model = "togethercomputer/llama-2-70b-chat"
    together_api_key = together.api_key 
    temperature = 0.1
    max_tokens = 1024

    class Config:
        extra = 'forbid'

    @root_validator()
    def validate_environment(cls, values):
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values
    
    @property
    def _llm_type(self):
        return "together"

    def _call(self, prompt, **kwargs):
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = output['output']['choices'][0]['text']
        return text

# Inicialización de la clase TogetherLLM
tog_llm = TogetherLLM(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.1,
    max_tokens=1024
)

# Definición de instrucciones del prompt
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """
Eres un asistente servicial, respetuoso y honesto. ...
"""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = f"{B_SYS}{new_system_prompt}{E_SYS}"
    return f"{B_INST}{SYSTEM_PROMPT}{instruction}{E_INST}"

# Generación del prompt
instruction = "CONTEXT:/n/n {context}. Pregunta: {question}. Answer:"
my_template = get_prompt(instruction)
llama_prompt = PromptTemplate(template=my_template, input_variables=["context", "question"])

# Funciones de utilidad para el procesamiento del texto
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    return '\n'.join(lines)

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])

def generate_response(uploaded_file, query_text):
    session_uuid = uuid.uuid4()
    
    if uploaded_file is not None:
        try:
            unique_filename = str(uuid.uuid4()) + "_" + uploaded_file.name
            file_path = os.path.join('data', unique_filename)
            
            # Guardar temporalmente el archivo cargado
            with tempfile.NamedTemporaryFile(dir='data', delete=False) as fp:
                fp.write(uploaded_file.getbuffer())
                temp_path = fp.name
            
            # Mover archivo a la ubicación deseada
            shutil.move(temp_path, file_path)
            st.success("El archivo se ha cargado correctamente.")
            
            # Carga y procesamiento de documentos
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            # Preparación de embeddings
            st.info("La inteligencia artificial está haciendo su magia... 🪄")
            model_name = "BAAI/bge-base-en"
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # Creación de Chroma DB
            db_name = f"./data/chroma_db_{session_uuid}.db"
            db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=db_name)
            
            # Creación de cadena de preguntas y respuestas (QA)
            qa = RetrievalQA.from_chain_type(
                llm=tog_llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
            
            # Consulta y respuesta
            llm_response = qa(query_text)
            return process_llm_response(llm_response)
        
        except Exception as e:
            st.error(f"Error al leer el archivo PDF: {e}")
            

# Verificar la existencia del directorio de datos
if not os.path.exists('data'):
    os.makedirs('data')

# Interfaz de usuario de Streamlit
uploaded_file = st.file_uploader('Sube un archivo', type='pdf')
query_text = st.text_input('Escribe tu pregunta aquí:', disabled=not uploaded_file)

result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('... 🤓'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.success(response)
    shutil.rmtree('./data')
