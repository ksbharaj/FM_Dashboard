import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import ScraperFC as sfc
import traceback
from google.cloud import storage
import os

def upload_to_gcs(bucket_name, object_name, local_file):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def scraperFC_FBRef():
    scraper = sfc.FBRef()
    try:
        lg_table = scraper.scrape_league_table(year=2023, league='EPL')
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        scraper.close()

    lg_table.sort_values(by='Squad', ascending=True, inplace=True)
    lg_table.reset_index(drop=True, inplace=True)

    file_path = '/tmp/league_table.csv'
    lg_table.to_csv(file_path, index=False)

    upload_to_gcs('gegenstats', 'league_table.csv', file_path)

    return 'gs://gegenstats/league_table.csv'