import os
import git
from tqdm import tqdm
import logging
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from util import *

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    # "Cookie": "" 
}


if __name__ == '__main__':
    # --------------------- load cve id ---------------------
    df = pd.read_csv('../data/data.csv')
    df = df[['cve']]
    cve_list = df.cve.unique()

    # --------- get vuln release time from cve page ---------
    result_list = []
    for cve in tqdm(cve_list):
        page = 'https://cve.mitre.org/cgi-bin/cvename.cgi?name='+cve
        res = requests.get(url=page,  headers=headers)
        time.sleep(5)   # Prevent frequent visits
        cvetime = re.search('<td><b>([0-9]{8})</b></td>', res.text).group(1)
        result_list.append([cve, cvetime])

    df = pd.DataFrame(result_list, columns=['cve', 'cvetime'])

    # --------- get vuln links and CWE from nvd page ---------
    result_list = []
    for cve in tqdm(cve_list):
        page = 'https://nvd.nist.gov/vuln/detail/'+cve
        try:
            links = []
            cwe = []

            res = requests.get(url=page, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')

            tbody = soup.find(attrs={'data-testid': "vuln-hyperlinks-table"}).tbody
            for tr in tbody.children:
                if isinstance(tr, NavigableString):
                    continue
                tds = tr.findAll('td')
                links.append(tds[0].a['href'])

            tbody = soup.find(attrs={'data-testid': "vuln-CWEs-table"}).tbody
            for tr in tbody.children:
                if isinstance(tr, NavigableString):
                    continue
                tds = tr.findAll('td')
                cwe.append((tds[0].text.replace('\n', ''), tds[1].text))

            result_list.append([cve, links, cwe])
        except Exception as e:
            logging.info(page + " ")

        time.sleep(5)   # Prevent frequent visits

    df2 = pd.DataFrame(result_list, columns=['cve', 'links', 'cwe'])
    df2 = df2.drop_duplicates(['cve']).reset_index(drop=True)

    # ---------------- load vuln description ----------------
    df3 = pd.read_csv("../data/cve_desc.csv", encoding='latin1')

    # ------------------------ merge ------------------------
    df4 = df.merge(df2[['cve', 'links', 'cwe']], how='left', on='cve').merge(df3, how='left', on='cve')
    df4.to_csv("../data/vuln_data_source.csv", index=False)

    # --------------------- tokenize cwe ---------------------
    df = pd.read_csv('../data/vuln_data_source.csv')
    df['cwe'] = df['cwe'].apply(eval)
    df['cwedesc'] = df['cwe'].apply(lambda items: ' '.join([item[1] for item in items]))
    df['cwedesc'] = df['cwedesc'].fillna('')
    df['cwedesc'] = df['cwedesc'].apply(lambda x: to_token(x))
    df.to_csv("../data/vuln_data.csv", index=False)
