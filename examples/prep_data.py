
from ../destruction_preprocess.py import prep_all

cities = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqaa']
for city in cities:
    prep_all(city, '')
