import os
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import GoogleDriveLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = ""

manualFolder = "1r0aF6cGeRa5GStGVzrJEn6sVUG66l4Db"

loader = GoogleDriveLoader(folder_id=manualFolder,
                          recursive=True,
                          credentials_path="./credentials.json",
                          token_path="./token.json",
                          file_types=["document"])
data = loader.load()

print (f'You have {len(data)} document(s) in your data')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

query = "¿Cuáles son los datos de facturación de Neoco?"
docs = db.similarity_search(query)

llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", verbose=True)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
