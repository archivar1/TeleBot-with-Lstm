import os
import telebot
from Predict_LSTM import predict_genre
bot = telebot.TeleBot('token') # Введите токен своего бота
about = 'В этого бота встроена нейросеть, которая определяет жанр музыки.\nДля определения отправьте аудиофайл'
@bot.message_handler(commands=['start'])
def startBot(message):
  first_mess = f"<b>{message.from_user.first_name} </b>, Привет!\n{about} "
  bot.send_message(message.chat.id, first_mess, parse_mode='html')

@bot.message_handler(content_types=['text'])
def get_text_message(message):
    if message.text =='/help':
        bot.send_message(message.chat.id , about)
    else:
        bot.send_message(message.chat.id, 'Сообщение не распознано, напишите /help для получение справки о боте')
@bot.message_handler(content_types=['audio'])
def get_audio_message(message):
    file_id = message.audio.file_id
    file_path = bot.get_file(file_id).file_path
    downloaded_file = bot.download_file(file_path)
    with open('audio.wav', 'wb') as f:
        f.write(downloaded_file)

    message_wait = bot.reply_to(message, 'Пожалуйста подождите, идет определение жанра...')
    genre = predict_genre('audio.wav')
    bot.delete_message(message.chat.id, message_id=message_wait.message_id)
    bot.reply_to(message, f'{genre}')
    os.remove('audio.wav')

bot.polling(none_stop=True, interval=1)