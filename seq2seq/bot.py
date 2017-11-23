"""
  1. get message
      return telegram.Update.message.chat_id,
             telegram.Update.message.text,
             telegram.Update.message.message_id      
      
  2. seq2seq learning
      call function,
      
  3. send message
      telegram.Bot.sendMessage(chat_id, text,
                               reply_to_message_id)
"""
import logging, random
import telegram

from datetime import datetime
from time import sleep

from telegram.error import NetworkError, Unauthorized
from telegram.ext import Updater

import CONFIG
import chatbot
import data_utils

# Model and Vocab dictionary.
vdict = data_utils.read_vocab_dict(vpath='./data/vocab')
model = chatbot.read_model()

# Global Variable: Telegram Token.
TOKEN = CONFIG.TOKEN
BOT = telegram.Bot(TOKEN)
lastMessageId = 0

# Enable Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def getText(Update):
  return Update.message.text

def getMessageId(Update):
  return Update.update_id

def getChatId(Update):
  return Update.message.chat.id

def getUserId(Update):
  return Update.message.from_user.id

def start(Update):
  return

def messageHandler(Update):
  global lastMessageId
  text = getText(Update)
  msg_id = getMessageId(Update)
  user_id = getUserId(Update)
  lastMessageId = msg_id
  msg_id = getMessageId(Update)
  #if lastMessageId != msg_id:
    #if text == '/stop': # Button function add by Bot-father
    #    break
    #  else:
  reply_text = chatbot.chat(text, model, vdict)
  BOT.sendMessage(user_id, reply_text)
  print('> send from ', msg_id, ':', text)
  print('> send to ', msg_id, ':', reply_text)
  # add logging to txt
  return

def error(bot, update, error):
  logger.warn('Update "%s" caused error "%s"' % (update, error))

def main():
  global lastMessageId
  Updates = BOT.getUpdates()
  if len(Updates) > 0:
    lastMessageId = Updates[-1].update_id
  while True:
    Updates = BOT.getUpdates(offset=lastMessageId)
    Updates = [Update for Update in Updates if Update.update_id > lastMessageId]
    for Update in Updates:
      messageHandler(Update)
    sleep(0.5)

if __name__ == '__main__':
  main()
