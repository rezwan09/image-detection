import random

from flask import Flask, request, jsonify
import boto3
#import json
from datetime import datetime, timedelta
import calendar
import time
from boto3.dynamodb.conditions import Key, Attr
import os
import our_model
import zipfile

#import io
#import base64

application = app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "test/"


@app.route('/')
def index():
    print("RUNNING: image-detection")
    return "Hello from image detection!"


@app.route('/detect/image', methods=['POST'])
def detect_image():
    # Image Detection code here
    # Update num of active users
    dynamodb = boto3.resource('dynamodb', region_name="us-west-2")
    table = dynamodb.Table("users_count")

    response = table.get_item(
        Key={
            'id': "1"
        }
    )
    num_logged_users = int(response["Item"].get("num_logged_users"))
    num_logged_users += 1
    print("num_logged_users = ", num_logged_users)
    response = table.put_item(
        Item={
            "id": "1",
            "num_logged_users": num_logged_users
        }
    )

    start_time = datetime.now()
    if request.files:
        image = request.files["image"]
        print("file name = ", image.filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

        # call the model
        time.sleep(20)
        #ret = model.detect()
    end_time = datetime.now()
    diff = end_time-start_time
    print(f"Single Image Detection took: {diff.total_seconds()} seconds")
    # Decrement 1 from number of active users
    response = table.get_item(
        Key={
            'id': "1"
        }
    )
    num_logged_users = int(response["Item"].get("num_logged_users"))
    num_logged_users -= 1
    print("num_logged_users = ", num_logged_users)
    response = table.put_item(
        Item={
            "id": "1",
            "num_logged_users": num_logged_users
        }
    )
    return str(random.randint(0,1))


@app.route('/detect/images', methods=['POST'])
def detect_images():
    # Image Detection code here
    # Update num of active users
    dynamodb = boto3.resource('dynamodb', region_name="us-west-2")
    table = dynamodb.Table("users_count")

    response = table.get_item(
        Key={
            'id': "1"
        }
    )
    num_logged_users = int(response["Item"].get("num_logged_users"))
    num_logged_users += 1
    print("num_logged_users = ", num_logged_users)
    response = table.put_item(
        Item={
            "id": "1",
            "num_logged_users": num_logged_users
        }
    )

    start_time = datetime.now()

    if request.files:
        images = request.files["images"]
        print("file name = ", images.filename)
        images.save(os.path.join(app.config["IMAGE_UPLOADS"], images.filename))
        with zipfile.ZipFile("test/cancer.zip", "r") as zip_ref:
            zip_ref.extractall("test/cancer")
        ret_classes = our_model.detect_images()

    end_time = datetime.now()
    diff = end_time - start_time
    print(f"Single Image Detection took: {diff.total_seconds()} seconds")
    # Decrement 1 from number of active users
    response = table.get_item(
        Key={
            'id': "1"
        }
    )
    num_logged_users = int(response["Item"].get("num_logged_users"))
    num_logged_users -= 1
    print("num_logged_users = ", num_logged_users)
    response = table.put_item(
        Item={
            "id": "1",
            "num_logged_users": num_logged_users
        }
    )

    return str(ret_classes)


if __name__ == "__main__":
    app.run(host='localhost', port=5001, debug=True)
    #app.run(host='0.0.0.0', port=8080)
    print('Server running with flask')
