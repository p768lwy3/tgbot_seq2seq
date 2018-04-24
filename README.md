# Telegram_chatbot
Using Keras and telegram API to build a chatbot by using seq2seq generator method. </br>
## Required Libraries:
1.  [Keras](https://keras.io): Building LSTM seq2seq generator model.
2.  [Jieba](https://github.com/fxsjy/jieba): Tokenizing Chinese Sentences.
3.  [Telegram](https://github.com/python-telegram-bot/python-telegram-bot): Python telegram API.
## Files:
1.  data_utils.py: Tokenizing and Vectorizing sentences and words
2.  train.py: Building and Traing the LSTM model
3.  chatbot.py: Chating in console(?)
4.  bot.py: Connecting with telegram
## Problems:
1.  Since the [dataset](https://inclass.kaggle.com/rtatman/the-national-university-of-singapore-sms-corpus/version/1) is not chatting, the bot cannot learn to chat.
2.  Since the dataset is Mandarin but not Cantonese, there should be some trouble.
3.  Since it is hard to tokenize and vectorize Chinese, the result of the bot is quite bad.
4.  ...
## Solution(?):
1.  Find a better dataset or wrap it from Facebook or Twitter.
2.  Find a better method to vectorize Chinese. May be LDA or Word2Vec?
## Result:

![Sample 1.](https://github.com/p768lwy3/Telegrambot/blob/master/pic/23244278_1655902084480773_7548481696032972095_n.jpg)

![Sample 2.](https://github.com/p768lwy3/Telegrambot/blob/master/pic/3319335_1655902041147444_591198137054595988_n.jpg)

![Sample 3.](https://github.com/p768lwy3/Telegrambot/blob/master/pic/23376484_1655901937814121_878269907131833199_n.jpg)

![Sample 4.](https://github.com/p768lwy3/Telegrambot/blob/master/pic/23380080_1655901867814128_8799978012659069703_n.jpg)

## Reference:
    Tao Chen and Min-Yen Kan (2013). Creating a Live, Public Short Message Service Corpus: The NUS SMS Corpus. Language Resources and Evaluation, 47(2)(2013), pages 299-355. URL: https://link.springer.com/article/10.1007%2Fs10579-012-9197-9

