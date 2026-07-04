# vigilant-guacamole
For my Embedded programming
# Edge Impulse - OpenMV FOMO Object Detection Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import sensor, image, time, ml, math, uos, gc, display, pyb
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.
lcd = display.SPIDisplay()
net = None
labels = None
min_confidence = 0.5
uart = pyb.UART(3, 9600)
uart.init(9600, bits=8, parity=None, stop=1)

def sent_asr(message):
    uart.write(message + "\n")
    print(f"Speak: {message}")


def recive_asr():
    data = uart.read()
    if data:
        try:
            recv_str = data.decode("utf-8").strip()
            print("Recive:", recv_str)
            return recv_str
        except Exception as e:
            print("Recive Fail:", e)
            return None
    '''
    if data is None:
        return None
    if data == "\n":
        try:
            recv_str = data.decode("utf-8").strip()
            print("Recive:", recv_str)
            return recv_str
        except Exception as e:
            print("Recive Fail:", e)
            return None
    '''

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

threshold_list = [(math.ceil(min_confidence * 255), 255)]

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l


clock = time.clock()
last_speak_time = 0
speak_cooldown = 5000  # 语音播报设置5秒冷却
while(True):
    clock.tick()
    is_fire = False
    is_smoke = False
    last_fire = False
    last_smoke = False
    current_time = pyb.millis()
    img = sensor.snapshot()

    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0: continue  # background class
        if len(detection_list) == 0: continue  # no detections for this class?

        # print("********** %s **********" % labels[i])
        label_name = labels[i] if i < len(labels) else f"class_{i}"

        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))

            if "fire" in label_name.lower():
                is_fire = True
            elif "smoke" in label_name.lower():
                is_smoke = True

            # print(f"x {center_x}\ty {center_y}\tscore {score}")
            img.draw_circle((center_x, center_y, 12), color=colors[i])


    # 只有状态变化 且 冷却已过 才播报
    if is_fire and not last_fire and (current_time - last_speak_time) > speak_cooldown:
        sent_asr("fire")
        last_speak_time = current_time

    if is_smoke and not last_smoke and (current_time - last_speak_time) > speak_cooldown:
        sent_asr("smoke")
        last_speak_time = current_time

    last_fire = is_fire
    last_smoke = is_smoke

    lcd.write(img)

    cmd = recive_asr()
    if cmd == "close":
        black_screen = image.Image(sensor.width(), sensor.height(), sensor.RGB565)
        black_screen.draw_rectangle(0, 0, sensor.width(), sensor.height(), color=0, fill=True)
        lcd.write(black_screen)
        sensor.sleep(True)
        while True:
            cmd = recive_asr()
            if cmd == "open":
                sensor.sleep(False)
                break
