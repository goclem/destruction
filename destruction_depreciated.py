
def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    regex = fr'^.*{city}/.*/{type}_{date}\.{ext}$'
    return regex

def extract(files:list, pattern:str=r'\d{4}-\d{2}-\d{2}') -> list:
    regex = re.compile(pattern)
    match = np.array([regex.search(file).group() for file in files])
    return match