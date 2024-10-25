import openai
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Set your OpenAI and Telegram bot tokens
openai.api_key = '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'
TELEGRAM_TOKEN = '7509744768:AAF9py0gQGdO61IecG_oPPkVtjs0gli8l3I'

# Function to handle user messages
def handle_message(update, context):
    user_message = update.message.text
    
    try:
        # Send the user's message to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        # Get the response text
        bot_response = response['choices'][0]['message']['content']
    except Exception as e:
        bot_response = "Maaf, ada masalah dalam memproses permintaanmu."

    # Send the response back to the user
    update.message.reply_text(bot_response)

# Function to start the bot
def start(update, context):
    update.message.reply_text("Halo! Aku adalah bot teman curhat dan bantu matematika. Silakan tanyakan apapun!")

# Main function to set up the bot
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
