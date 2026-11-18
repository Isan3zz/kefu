from app.config import settings
from app.db import DB
from app.rag import RAG
if __name__=='__main__': print(RAG(DB(settings.sqlite_url)).build('knowledge'))
