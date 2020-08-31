from flask import Flask, request, jsonify, abort
from flask_restful import reqparse, abort, Api, Resource

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    MemberJoinedEvent, MemberLeftEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, SpacerComponent, IconComponent, ButtonComponent,
    SeparatorComponent, QuickReply, QuickReplyButton,
    ImageSendMessage)

import imdb
import json
import errno
import os
import sys
import tempfile

IMG_SIZE = 224
CHANNELS = 3

PREDICT_LABELS = [
    'Action',
    'Adventure',
    'Animation',
    'Biography',
    'Comedy',
    'Crime',
    'Drama',
    'Family',
    'Fantasy',
    'Game-Show',
    'History',
    'Horror',
    'Music',
    'Musical',
    'Mystery',
    'News',
    'Romance',
    'Sci-Fi',
    'Sport',
    'Thriller',
    'War',
    'Western',
]

model = tf.keras.models.load_model(
    "model_20200829.h5", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

app = Flask(__name__)
api = Api(app)

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# function for create tmp dir for download content


def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


# create tmp dir for download content
make_static_tmp_dir()


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    if text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' +
                                    profile.display_name),
                    TextSendMessage(text='Status message: ' +
                                    str(profile.status_message))
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))

    else:
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.message.text))

# Other Message Type


@handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)

    predict_message = json.dumps(predict(
        os.path.join('static', 'tmp', dist_name)))

    line_bot_api.reply_message(
        event.reply_token, [
            # TextSendMessage(text='Save content.'),
            # TextSendMessage(text=request.host_url + \
            #                 os.path.join('static', 'tmp', dist_name)),
            TextSendMessage(text=predict_message)
        ])


@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn (default)".format(
        version
    )
    return message


def predict(image_path, isUrl = False):
    if isUrl:
        img_path = tf.keras.utils.get_file(fname=next(tempfile._get_candidate_names()), origin=image_url)

    img = keras.preprocessing.image.load_img(
        img_path, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array/255
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Generate prediction
    predict_value = model.predict(img_array)
    prediction = (predict_value > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction = prediction[prediction == 1].index.values

    os.remove(img_path)

    response = {}
    response['predict_genres'] = [
        f'{PREDICT_LABELS[p]}: {predict_value.tolist()[0][p]:.2f}' for p in prediction]

    return response


class Imdb(Resource):
    def get(self, title_id):
        print(f"got title_id {title_id}")

        # creating instance of IMDb
        ia = imdb.IMDb()
        movie = ia.get_movie(int(title_id[2:]))

        response = {}
        response['id'] = title_id
        response['title'] = movie['title']
        response['genres'] = movie['genres']

        movie_img_url = movie['full-size cover url']

        predict_genres = predict(movie_img_url, isUrl = True)

        #Merge Dict
        response = {**response, **predict_genres}

        return response


##
# Actually setup the Api resource routing here
##
api.add_resource(Imdb, '/imdb/<title_id>')
# api.add_resource(Todo, '/todos/<todo_id>')
