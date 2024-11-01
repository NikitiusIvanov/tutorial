import os
import io
import sys
import logging
import asyncio
from aiohttp import web
import numpy as np

# Aiogram imports
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    Message,
    KeyboardButton,
    ReplyKeyboardMarkup,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from aiogram.utils.keyboard import (
    ReplyKeyboardBuilder,
    InlineKeyboardBuilder
)

# Google Generative AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Image

# SQLAlchemy imports
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine
)
from sqlalchemy.sql import text


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
try:
    IS_LOCAL_DEBUG = bool(os.getenv('IS_LOCAL_DEBUG'))
except:
    IS_LOCAL_DEBUG = False
logging.info(f'IS_LOCAL_DEBUG: {IS_LOCAL_DEBUG}')

# Bot token can be obtained via https://t.me/BotFather
BOT_TOKEN = os.getenv("BOT_TOKEN")
# Webserver settings
# bind localhost only to prevent any external access
WEB_SERVER_HOST = "0.0.0.0"
# Port for incoming request from reverse proxy. Should be any available port
WEB_SERVER_PORT = 8080
# Path to webhook route, on which Telegram will send requests
WEBHOOK_PATH = "/webhook"
# Base URL for webhook will be used to generate webhook URL for Telegram,
# in this example it is used public DNS with HTTPS support
BASE_WEBHOOK_URL = os.getenv('BASE_WEBHOOK_URL')

# Settings for vertexai initialization
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')

vertexai.init(project=PROJECT_ID, location=REGION)
generation_config = {
    'temperature': 0,
}

model = GenerativeModel("gemini-1.5-pro-002")
####################### Set prompt for tasks #######################
PROMPT = """
You are a helpful AI assistant that helps people collect data about their diets
by food photos that they send to you.
Recognize food in this picture and give your estimation of the:
* total amount of calories in kcal,
* mass in grams,
* fat in grams,
* carbohydrates in grams


If you recognize there is no food in the photo
write your answer by following format and nothing more:


no food


If you recognize some food on the photo use low-high borders
for you estimation of the nutritional facts
and write your answer by following format and nothing more:


dish_name: apple pie
calories: 230, 240
mass: 340, 350
fat: 5.0, 5.5
carb: 22, 25
protein: 24, 25
"""


####################### Data processing #######################
DATA_PROCESSING_CHAPTER = None

def text_from_nutrition_facts(
    nutrition_facts: dict[str, str|float],
    is_saved: bool=False,
) -> str:
    text = (
        '*Here is my estimation of the nutrition facts about your photo:*\n'
        f'🍽 Dish name: *{nutrition_facts["dish_name"]}*\n'
        f'🧮 Total calories: *{round(nutrition_facts["calories"], 2)}* kcal\n'
        f'⚖️ Total mass: *{round(nutrition_facts["mass"], 2)}* g\n'
        f'🍖 Proteins mass: *{round(nutrition_facts["protein"], 2)}* g\n'
        f'🍬 Carbohydrates mass: *{round(nutrition_facts["carb"], 2)}* g\n'
        f'🧈 Fats mass: *{round(nutrition_facts["fat"], 2)}* g'
    )
    if is_saved == True:
        text+=(
            '\n✅ Saved to your meals'
        )
    
    return text.replace('.', '\.')


def response_to_dict(
    response: str,
) -> dict[str, list[float]] | str:
    """
    transform string 
    "dish_name: apple pie
    calories: 15, 25
    mass: 100, 150
    fat: 0.3, 0.5
    carb: 2, 4
    protein: 2, 3"
    -> to dict = {
        calories: 20,
        mass: 125,
        fat: 0.4,
        carb: 3,
        protein: 2.5,
        name: apple pie,
    }
    """
    if response.text == 'no food':
        return response.text
    else:
        try:
            result = {
                row.split(': ')[0]: np.mean(
                    [
                        round(float(row.split(': ')[1].split(',')[0]), 2),
                        round(float(row.split(': ')[1].split(',')[1]), 2)
                    ]
                )
                for row in response.text.strip('\n').split('\n')[1:]
            }
            result['dish_name'] = (
                response.text
                .strip('\n')
                .split('\n')[0]
                .split(': ')[1]
            )
            assert set(
                [
                    'dish_name', 'calories', 'mass', 'fat', 'carb', 'protein'
                ]
            ) == set(result.keys())
        except:
            result = 'not correct result'
    return result


def message_lenght(message_text: str | None):
    """Returns lenght of the recieved message
    """
    if message_text is not None:
        return len(message_text)
    else:
        return None


####################### Telegram Bot Handlers #######################
TELEGRAM_BOT_LOGIC_CHAPTER = None

# All handlers should be attached to the Router
router = Router()

class Form(StatesGroup):
    chat_id = State()
    photo_ok = State()
    nutrition_ok = State()
    nutrition_facts = State()
    username = State()
    first_name = State()
    last_name = State()
    user_id = State()
    edit_request = State()
    key_to_edit = State()
    new_value = State()
    edit_daily_goal = State()
    statistics = State()


def build_reply_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text='🍽 Recognize nutrition'),
        KeyboardButton(text='📝 Edit My daily goal'),
        KeyboardButton(text='📊 Get today\'s statistics'),
    )
    builder.adjust(2)

    return builder.as_markup()


def build_inline_keyboard(is_saved: bool=False) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    if is_saved == False:
        builder.row(
                InlineKeyboardButton(text='Edit name', callback_data='name'),
                InlineKeyboardButton(text='Edit mass', callback_data='mass')  
        )
        builder.row(
            InlineKeyboardButton(text='Edit calories', callback_data='calories'),
            InlineKeyboardButton(text='Edit proteins', callback_data='protein')
        )
        builder.row(
            InlineKeyboardButton(text='Edit carbs', callback_data='carb'),
            InlineKeyboardButton(text='Edit fats', callback_data='fat')
        )
    
        builder.row(
            InlineKeyboardButton(text='Save to my meals', callback_data='Save to my meals')
        )
    else:
        builder.row()
    return builder.as_markup()


@router.message(CommandStart())
async def welcome(
    message: Message,
    session: AsyncSession,
) -> None:
    first_name = message.from_user.first_name

    await message.answer(
        text=(
            f'👋 *Hey, {first_name}!* \n'
            'I\'m a helpful AI bot 🤖.'
            'Send me a photo 📸 of your food 🍽 \n'
            'and I\'ll recognize nutritional facts about it'
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_reply_keyboard()
    )

    # await check_user_exist(message=message, session=session)


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    # Send a response to the user
    await message.answer(
        "⚡️ Thank you for sending the photo! \n"
        "⚙️ It in processing, please wait your results",
        reply_markup=build_reply_keyboard()
    )

    await message.bot.send_chat_action(
        message.chat.id, 
        action=ChatAction.TYPING
    )

    # Get the largest available photo size
    photo_file_id = message.photo[-1].file_id
    
    # Download the photo as bytes
    bytes = io.BytesIO()
    photo_file = await message.bot.download(
        photo_file_id, 
        destination=bytes
    )

    img = Image.from_bytes(photo_file.read())

    request_parts = [PROMPT, img]
    
    response = model.generate_content(
        request_parts,
        generation_config=generation_config
    )
    result = response_to_dict(response)

    if result == 'no food' or result == 'not correct result':
        await message.answer(
            text=(
                '😔 Sorry I can not recognize food in your photo\n'
                '🙏 Please try once again'
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_reply_keyboard()
        )
    else:
        nutrition_facts = result

        text=text_from_nutrition_facts(nutrition_facts=nutrition_facts)
        
        await state.update_data(nutrition_facts=nutrition_facts)
        await state.update_data(chat_id=message.chat.id)
        await state.update_data(username=message.chat.username)
        await state.update_data(first_name=message.from_user.first_name)
        await state.update_data(last_name=message.from_user.last_name)
        await state.update_data(user_id=message.from_user.id)
        
        await message.answer(
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=build_inline_keyboard(),
        )


async def on_startup(bot: Bot) -> None:
    # If you have a self-signed SSL certificate, then you will need to send a public
    # certificate to Telegram so here we use GC servers and thats works without it
    await bot.set_webhook(f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}")


async def on_local_startup(bot: Bot) -> None:
    # delete webhook on local debugging
    await bot.delete_webhook()


def main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()
    # ... and all other routers should be attached to Dispatcher
    dp.include_router(router)

    # Register startup hook to initialize webhook
    dp.startup.register(on_startup)

    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # Create aiohttp.web.Application instance
    app = web.Application()

    # Create an instance of request handler,
    # aiogram has few implementations for different cases of usage
    # In this example we use SimpleRequestHandler which is designed to handle simple cases
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot
    )
    # Register webhook handler on application
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    # Mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # And finally start webserver
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


async def local_main() -> None:
    # Initialize Bot instance
    bot = Bot(BOT_TOKEN)
    
    # And the run events dispatching
    dp = Dispatcher()

    dp.include_router(router)

    # Register on startup function
    dp.startup.register(on_local_startup)

    await dp.start_polling(bot)


if __name__ == "__main__":
    if IS_LOCAL_DEBUG:
        asyncio.run(local_main())
    else:
        main()
