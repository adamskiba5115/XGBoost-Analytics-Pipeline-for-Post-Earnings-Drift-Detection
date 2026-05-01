import os
from sqlalchemy import create_engine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
DATABASE_URL = f'sqlite:///{DB_PATH}'


engine = create_engine(DATABASE_URL)