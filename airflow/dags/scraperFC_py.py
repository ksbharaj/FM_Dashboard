import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import ScraperFC as sfc
import traceback

import warnings
import uuid

def scraperFC_FBRef():
    scraper = sfc.FBRef()
    try:
        # Scrape the table
        lg_table = scraper.scrape_league_table(year=2023, league='EPL')
    except:
        # Catch and print any exceptions. This allows us to still close the
        # scraper below, even if an exception occurs.
        traceback.print_exc()
    finally:
        # It's important to close the scraper when you're done with it. Otherwise,
        # you'll have a bunch of webdrivers open and running in the background.
        scraper.close()
        
    lg_table.sort_values(by='Squad', ascending=True, inplace=True)

    lg_table.reset_index(drop=True, inplace=True)

    return (lg_table)