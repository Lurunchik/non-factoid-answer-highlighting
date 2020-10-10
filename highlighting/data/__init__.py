import pathlib

DATA_FOLDER = pathlib.Path(__file__).parent.absolute()

PRETRAINED_MODEL_FOLDER = DATA_FOLDER / 'nfl6_pretrained'
ANTIQUE_IDS_PATH = DATA_FOLDER / 'antique_ids.txt'
STUDY_QUERIES_PATH = DATA_FOLDER / 'study_queries_from_chiir2019.csv'
