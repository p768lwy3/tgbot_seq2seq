from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
# https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#yahoo-finance-quotes
# import quandl
import re
import requests

# html_parser and headers
HTML_PARSER = "html.parser"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}

# List of constituent stocks.
# better to have a way to update the list automatically.
constituent = ['00001', '00002', '00003', '00004', '00005', '00006', 
  '00011', '00012', '00016', '00017', '00019', '00023', '00027', '00066',
  '00083', '00101', '00135', '00144', '00151', '00175', '00267', '00293', 
  '00386', '00388', '00688', '00700', '00762', '00823', '00836', '00857', 
  '00883', '00939', '00941', '00992', '01038', '01044', '01088', '01109',
  '01113', '01299', '01398', '01928', '02018', '02318', '02319', '02388',
  '02628', '03328', '03988']

## for testing, getting HSI.
class Index(object):
  def __init__(self):
    self.url = 'https://hk.finance.yahoo.com/quote/%5EHSI?p=^HSI'
    self.index = 0

  def get(self):
    list_req = requests.get(self.url, headers=headers)
    if(list_req.status_code == requests.codes.ok):
      soup = BeautifulSoup(list_req.content, HTML_PARSER)
      index = soup.find('span',attrs={'class':'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
      return index

  def get_difference(self):
    index_now = self.get()
    diff = self.index - index_now()
    self.index = index_now()   
    return diff

## doing TA with the data
class Analysis(object):
  def __init__(self, code):
    self.code = code
    path_dataframe = 'data/' + ''.join([i for i in code if i.isdigit()]) + '.csv'
    self.data = pd.read_csv(path_dataframe)
    self.data = self.data.iloc[::-1] # re-order the data

  def real_time_price(self):
    code = self.code[1:] + '.HK'
    url = 'https://hk.finance.yahoo.com/quote/{0}?p={0}'.format(code)
    list_req = requests.get(url, headers=headers)
    if(list_req.status_code == requests.codes.ok):
      soup = BeautifulSoup(list_req.content, HTML_PARSER)
      self.price = soup.find('span', attrs={'class':'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
      self.data.loc[self.data.shape[0], 'adjClose'] = self.price

  def ma50(self):
    self.data['ma50'] = self.data['adjClose'].rolling(window=50).mean()

  def ma200(self):
    self.data['ma200'] = self.data['adjClose'].rolling(window=200).mean()

  def ma50200_trigger(self):
    self.ma50(); self.ma200()
    self.data['ma50-200'] = self.data['ma50'] - self.data['ma200']
    i = len(self.data)-1
    if self.data['ma50-200'][i] > 0 and self.data['ma50-200'][i-1] < 0:
      trigger = 'up'
    elif self.data['ma50-200'][i] < 0 and self.data['ma50-200'][i-1] > 0:
      trigger = 'down'
    else:
      trigger = 'none'
    return trigger

def get_data():
  # Webb-site Who's Who
  for code in constituent:
    starttime = datetime.now()
    path_dataframe = 'data/' + ''.join([i for i in code if i.isdigit()]) + '.csv'
    data = pd.read_csv(path_dataframe)
    last_date = datetime.strptime(data.loc[0, 'atDate'], '%Y-%m-%d')
    today = datetime.now()
    if last_date.year == today.year and last_date.month == today.month and last_date.day == today.day:
      print('Stock. {0} had been updated'.format(code))
    else:
      try:
        print('Start getting {0}'.format(code))
        url = "https://webb-site.com/dbpub/hpu.asp?s=datedn&sc={0}".format(code)
        list_req = requests.get(url, headers=headers)
        if(list_req.status_code == requests.codes.ok):
          soup = BeautifulSoup(list_req.content, HTML_PARSER)
          href_tag = soup.find('a', href=re.compile(r'pricesCSV\.asp\?i=\d+'))
          url = 'https://webb-site.com/dbpub/{0}'.format(href_tag['href'])
          data = pd.read_csv(url)
          path = 'data/' + ''.join([i for i in code if i.isdigit()]) + '.csv'
          data.to_csv(path_or_buf=path, header=True, index=True)
          print('Finish getting {0}.'.format(code))
          print('Time cost: {0}.'.format(str(datetime.now()-starttime)))
      except:
        print('Cannot get {0} from the website.'.format(code))
    
  '''
  # yahoo-finance
  start = datetime(2010, 1, 1)
  end = datetime(2017, 8, 15)
  for code in constituent:
    target = code[1:] + '.HK'
    print('Start getting {0}'.format(code))
    try:
      data = web.get_data_yahoo(target, start, end)
      path = 'data/' + ''.join([i for i in code if i.isdigit()]) + '.csv'
      data.to_csv(path_or_buf=path, header=True, index=True)
      print('Finish getting {0}'.format(code))
    except:
      print('Cannot get {0}'.format(code))
  '''
  '''
  # quandl
  # Using quandl to get data.
  # convert the names
  for i in range(len(constituent)):
    constituent[i] = 'HKEX/' + constituent[i]
  
  counter = 0
  for code in constituent:
    if counter >= 20:
      # Since qunadl API only arrow 20 calls per 10 minuters
      print('Sleep for 10 minute')
      time.sleep(600)
      counter = 0
      print('Wake up!')
    print('Start getting {0}'.format(code))
    data = quandl.get(code) # return dataframe
    path = 'data/' + ''.join([i for i in code if i.isdigit()]) + '.csv'
    data.to_csv(path_or_buf=path, header=True, index=True)
    print('Finish getting {0}'.format(code))
    counter = counter + 1
  '''

def analysis():
  result = []
  for code in constituent:
    print('Stock no: {0}'.format(code))
    analysis = Analysis(code)
    ## append real_time_price to the dataframe
    analysis.real_time_price()
    result.append((code, analysis.ma50200_trigger()))
  # return: a list of truples which [0] is the stock no, [1] is the status of ma50/200
  return result 
    
def close_price():
  end = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 16, 0)
  if datetime.now() >= end:
    get_data(constituent)

def main():
  #get_data()
  #analysis()
  #close_price()
  pass

if __name__ == '__main__':
  main()
