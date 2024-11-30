#frist u need set your screen to 2560 * 1600 
#首先设置屏幕分辨率，针对瓦建议使用1600p 即2560 *  1600
#pip install win32gui(use for screen shot)
#pyautogui was not used here


from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import win32gui, win32ui, win32con, win32api
import pyautogui

global n,s

frame = 1
scale_percent = 50 
n = 0
s = 720#int(input())

def capture_screenshot(width, height):
    # get your screen
    hwin = win32gui.GetDesktopWindow()
    left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
    top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    # get bit map
    bmp_info = bmp.GetInfo()
    bmp_bits = bmp.GetBitmapBits(True)
    # use Pillow open the img 
    image = Image.frombuffer(
        "RGB",(bmp_info["bmWidth"], bmp_info["bmHeight"]),bmp_bits,
        "raw","BGRX",
        0, 1
    )

    # release the memory
    memdc.DeleteDC()
    win32gui.DeleteObject(bmp.GetHandle())

    return image
if __name__ == '__main__':
    # Load a model
    model = YOLO(model=r'D:\pycharm\open-cv\ultralytics-main\runs\train\exp15\weights\best.pt')
    # use cv2 to show your results
    print("请输入分辨率 1600 1080 720")
    if s == 1080:
        width, height = 1920, 1080
    elif s == 1600:
        width, height = 2560, 1600
    elif s == 720:
        width, height = 1280, 720
    while True:
        n+=1
        x,y = pyautogui.position()
        screenshot = capture_screenshot(width, height)
        image = np.array(screenshot)
        # CV2 img need BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        w = int(image.shape[1] * scale_percent / 100)
        h = int(image.shape[0] * scale_percent / 100)
        dim = (w, h)
        results = model.predict(image,conf=0.5)
        for i in results:
            for box in i.boxes:
                x_new = (int(box.xyxy[0][0])+int(box.xyxy[0][2]))/2
                y_new = (int(box.xyxy[0][3])+int(box.xyxy[0][1]))/2

                cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 1)
                cv2.putText(image, f"{i.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Screenshot", resized_image)
        # cv2.imwrite(rf'C:\Users\18538\Desktop\screenshot\11-27screenshot{n}.png', resized_image)
        # set your Frame rate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("over screenshot")
            break
        cv2.waitKey(frame)  # 截图延迟性能



    cv2.destroyAllWindows()
