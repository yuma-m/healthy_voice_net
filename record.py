#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import wave
import datetime

import pyaudio
import RPi.GPIO as GPIO
import requests

from test_nn import AudioNetTester


# 録音デバイス番号
DEVICE = 2

# LEDとタクトスイッチのピン
SWITCH_PIN = 21
GREEN_PIN = 12
YELLOW_PIN = 16
RED_PIN = 20

# IDCFクラウドのIPアドレス
IDCF_IP = 'XXX.XXX.XXX.XXX'
# MeshblueのTriggerのUUIDとToken
UUID = "trigger-uuid"
TOKEN = "trigger-token"

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)


def check_devices():
    u'''オーディオデバイス番号の確認'''
    p = pyaudio.PyAudio()
    count = p.get_device_count()
    devices = []
    for i in range(count):
        devices.append(p.get_device_info_by_index(i))

    for i, dev in enumerate(devices):
        print (i, dev['name'])


def record_wav():
    u'''wavファイルを録音する'''
    FORMAT = pyaudio.paInt16
    CHANNELS = 1        # モノラル
    RATE = 44100        # サンプルレート
    CHUNK = 2**13       # データ点数
    RECORD_SECONDS = 3  # 録音する時間の長さ
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    WAVE_OUTPUT_FILENAME = "voice/%s.wav" % (now)

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=DEVICE,  # デバイスのインデックス番号
                        frames_per_buffer=CHUNK)
    print ("recording...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return WAVE_OUTPUT_FILENAME


def blink_led(normal, cold, sleepy):
    u'''診断結果をLEDで表示'''
    if normal > cold and normal > sleepy:
        GPIO.output(GREEN_PIN, GPIO.HIGH)
        time.sleep(5.0)
        GPIO.output(GREEN_PIN, GPIO.LOW)
    elif cold > sleepy:
        GPIO.output(RED_PIN, GPIO.HIGH)
        time.sleep(5.0)
        GPIO.output(RED_PIN, GPIO.LOW)
    else:
        GPIO.output(YELLOW_PIN, GPIO.HIGH)
        time.sleep(5.0)
        GPIO.output(YELLOW_PIN, GPIO.LOW)


def send_data(normal, cold, sleepy, wav_file):
    u'''IDCFクラウドにデータを送信する'''
    print("Send trigger")
    trigger_url = "http://%s/data/%s" % (IDCF_IP, UUID)
    headers = {
        "meshblu_auth_uuid": UUID,
        "meshblu_auth_token": TOKEN
    }
    payload = {
        'trigger': 'on', 'normal': normal,
        'cold': cold, 'sleepy': sleepy
    }
    requests.post(trigger_url, headers=headers, data=payload)

    print("Sending data finished")


def main():
    last_pin_status = 0
    tester = AudioNetTester()
    print("Waiting")

    while True:
        pin_status = GPIO.input(SWITCH_PIN)

        if last_pin_status == 1 and pin_status == 0:
            time.sleep(0.1)
            GPIO.output(GREEN_PIN, GPIO.HIGH)
            wav_file = record_wav()
            GPIO.output(GREEN_PIN, GPIO.LOW)

            normal, cold, sleepy = tester.test(wav_file)

            blink_led(normal, cold, sleepy)

            send_data(normal, cold, sleepy, wav_file)

            print("Waiting")

        last_pin_status = pin_status
        time.sleep(0.1)

    GPIO.cleanup()


if __name__ == '__main__':
    main()
