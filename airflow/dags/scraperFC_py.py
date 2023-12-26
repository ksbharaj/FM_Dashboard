import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import ScraperFC as sfc
import traceback
from google.cloud import storage
import pickle

def upload_to_gcs(bucket_name, object_name, local_file):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def scraperFC_Understat():
    scraper = sfc.Understat()
    try:
        lg_table = scraper.scrape_league_table(year=2024, league='EPL')
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        scraper.close()

    understat_table = lg_table[['Team', 'W', 'D', 'L', 'G', 'GA', 'xG', 'xGA', 'NPxG',
                                'NPxGA', 'PPDA', 'OPPDA', 'PTS', 'xPTS']]
    understat_table['season'] = 2324
    understat_table['league'] = 'ENG-Premier League'

    understat_table.loc[understat_table['Team'] == 'Manchester United', 'Team'] = 'Manchester Utd'
    understat_table.loc[understat_table['Team'] == 'Newcastle United', 'Team'] = 'Newcastle Utd'
    understat_table.loc[understat_table['Team'] == 'Tottenham Hotspur', 'Team'] = 'Tottenham'
    understat_table.loc[understat_table['Team'] == 'Wolverhampton Wanderers', 'Team'] = 'Wolves'
    understat_table.loc[understat_table['Team'] == 'Nottingham Forest', 'Team'] = "Nott'ham Forest"
    understat_table.loc[understat_table['Team'] == 'Leicester', 'Team'] = 'Leicester City'
    understat_table.loc[understat_table['Team'] == 'Leeds', 'Team'] = 'Leeds United'
    understat_table.loc[understat_table['Team'] == 'Sheffield United', 'Team'] = "Sheffield Utd"
    understat_table.loc[understat_table['Team'] == 'Luton', 'Team'] = 'Luton Town'
    understat_table.loc[understat_table['Team'] == 'Norwich', 'Team'] = 'Norwich City'

    understat_table.sort_values(by=['league', 'season', 'Team'], ascending=True, inplace=True)
    understat_table.reset_index(drop=True, inplace=True)

    understat_table.rename(columns={'Team': 'TEAM_NAME', 'W': 'TEAM_WINS', 'D': 'TEAM_DRAWS', 'league': 'COMPETITION',
                                    'L': 'TEAM_LOSSES', 'xPTS': 'TEAM_XPTS', 'PTS': 'TEAM_PTS', 'season': 'SEASON',
                                    'NPxGA': 'NPXG_AGAINST', 'xGA': 'XG_AGAINST'}, inplace=True)

    understat_table.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)

    # file_path = '/tmp/league_table_understat.csv'
    # understat_table.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'league_table_understat.csv', file_path)

    return pickle.dumps(understat_table)