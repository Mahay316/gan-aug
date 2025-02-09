from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import sys

import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import os


def send_email(to_email, password, subject, body=''):
    # sender configuration
    from_email = 'pengfeizhao316@gmail.com'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Date'] = formatdate(localtime=True)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, [to_email], msg.as_string())
    except Exception as e:
        print(f'Mail Sending Failed: {e}')


if os.environ.get('EMAIL_PASSWORD') is not None:
    password = os.environ.get('EMAIL_PASSWORD')
else:
    raise AttributeError('No password is provided. Try providing one with the environment variable EMAIL_PASSWORD')

if os.environ.get('OBFS_TYPE') is not None:
    obfs_type = os.environ.get('OBFS_TYPE')
else:
    raise AttributeError('No obfuscator type is provided. Try providing one with the environment variable OBFS_TYPE')

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless if you don't need a UI
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--proxy-server=socks5://localhost:9050')

# Set up the Chrome driver
service = Service('/usr/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

website_list_file = 'top-2k.csv'
chunk_size = 1000
idle_duration = 2  # in seconds

target_count = 2000
email_lead = 20
to_email = '1422411405@qq.com'

# for statistics
successCnt = 0
totalCnt = 0
for chunk in pd.read_csv(website_list_file, chunksize=chunk_size):
    for idx, row in chunk.iterrows():
        website = row['url']
        totalCnt = totalCnt + 1
        try:
            # Visit the website
            driver.get('http://' + website)
            time.sleep(idle_duration)

            # Get the page title and current URL
            page_title = driver.title
            current_url = driver.current_url

            successCnt = successCnt + 1
            print(f"Number: {totalCnt}, Website: {current_url}, Title: {page_title}, Stat: {successCnt}/{totalCnt}")
            sys.stdout.flush()

        except Exception as e:
            print(f"Failed to access {website}")
            sys.stdout.flush()

        if totalCnt == target_count - email_lead:
            send_email(to_email,
                       password,
                       subject='Data Collection Progress Report',
                       body=f'{obfs_type}: {successCnt} / {totalCnt}'
                       )

print('Request finished')
print(f'Successful: {successCnt}/{totalCnt}')

# Clean up
driver.quit()
