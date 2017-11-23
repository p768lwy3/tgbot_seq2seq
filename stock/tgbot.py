from datetime import datetime, time
import logging
import random
import requests
import telegram
from telegram.error import NetworkError, Unauthorized
from telegram.ext import Updater, CallbackQueryHandler, CommandHandler, ConversationHandler, Filters, Job, MessageHandler, Handler
from time import sleep

# for test. 
from datetime import datetime

import stockcrawl

# telegram token
TOKEN = "424469480:AAFSZ7hRbUgS79-_mqzR9-aB9liBSPYRQpM"
update_id = None

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

# Command Function:
# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def analysis(bot, update):
  while datetime.now().hour >= 9 and datetime.now().hour <= 16:
  # stock market time
    if datetime.now().minute % 30 == 0: 
    # run the analysier 30 min once time.
      update.message.reply_text('Start...')
      text_time = '依家時間係: {0}:{1}:{2}'.format(datetime.now().hour, datetime.now().minute, datetime.now().second)
      update.message.reply_text(text_time)
      results = stockcrawl.analysis()
      for result in results:
        if result[1] == 'up':
          text = '股票{0}的50天平均線升穿200天平均線'.format(result[0])
          update.message.reply_text(text)
        elif result[1] == 'down':
          text = '股票{0}的50天平均線跌穿200天平均線'.format(result[0])
          update.message.reply_text(text)
      update.message.reply_text('完！')
      text_time = '依家時間係: {0}:{1}:{2}'.format(datetime.now().hour, datetime.now().minute, datetime.now().second)
      update.message.reply_text(text_time)
      time.sleep(300)
  else:
    if datetime.now().hour < 9:
      update.message.reply_text('未開市!')
    if datetime.now().hour > 16:
      update.message.reply_text('市已收!')
    return

def update_dataset(bot, update):
  update.message.reply_text('開始更新!')
  stockcrawl.get_data()
  update.message.reply_text('完成更新!')

#######################################################################################
###################################################################### for testing use:
#######################################################################################

def test(bot, update):
  update.message.reply_text('測試開始!')
  results = stockcrawl.analysis()
  for result in results:
    if result[1] == 'up':
      text = '股票{0}的50天平均線升穿200天平均線'.format(result[0])
      update.message.reply_text(text)
    elif result[1] == 'down':
      text = '股票{0}的50天平均線跌穿200天平均線'.format(result[0])
      update.message.reply_text(text)
    elif result[1] == 'none':
      text = '股票{0}沒有表現'.format(result[0])
      update.message.reply_text(text)

def get(bot, update):
  index = stockcrawl.Index()
  result = index.get()
  text = '恆指現在是{0}'.format(result)
  update.message.reply_text(text)

def test2(bot, update):
  index = stockcrawl.Index()
  #while datetime.now().hour >= 9 and datetime.now().hour <= 16:
  # stock market time
  if datetime.now().minute % 2 == 0: 
  # run the analysier 10 min once time.
    update.message.reply_text('開始測試!')
    result = index.get().replace(",",""); diff = str(float(index.index) - float(result))
    text = '恆指現在是{0},升/跌{1}'.format(result, diff)
    update.message.reply_text(text)
    index.index = str(result)
  ## cannot stop at the time period    
  else:
    if datetime.now().hour < 9:
      bot.send_message(job.context, '未開市!')
    if datetime.now().hour > 16:
      bot.send_message(job.context, '市已收!')

#######################################################################################
#######################################################################################
#######################################################################################

def error(bot, update, error):
  logger.warn('Update "%s" caused error "%s"' % (update, error))

def main():
  updater = Updater(TOKEN) # Create the EventHandler and pass it your bot's token.
  dp = updater.dispatcher # Get the dispatcher to register handlers

  # Command
  dp.add_handler(CommandHandler("start", analysis))
  dp.add_handler(CommandHandler("update", update_dataset))
  #dp.add_handler(CommandHandler("test", test))
  #dp.add_handler(CommandHandler("test2", test2))
  
  # log all errors
  dp.add_error_handler(error)

  # Start the Bot
  updater.start_polling()

  # Run the bot until you press Ctrl-C or the process receives SIGINT,
  # SIGTERM or SIGABRT. This should be used most of the time, since
  # start_polling() is non-blocking and will stop the bot gracefully.
  updater.idle()

if __name__ == '__main__':
  main()
