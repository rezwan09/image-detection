from flask import Flask, request, jsonify
import boto3
import json
from decimal import Decimal
from datetime import datetime, timedelta
import calendar
import time
from boto3.dynamodb.conditions import Key, Attr
import os
import io
import base64

application = app = Flask(__name__)


@app.route('/')
def index():
    print("RUNNING: image-detection")
    return "Hello World!"


@app.route('/detect/image', methods=['POST'])
def get_settings():
    # Image Detection code here
    return None


if __name__ == "__main__":
    #app.run(host='localhost', port=5001, debug=True)
    app.run(host='0.0.0.0', port=8080)
    print('Server running with flask')
