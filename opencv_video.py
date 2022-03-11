import dlib
import cv2

# 匯入影片
cap = cv2.VideoCapture("video.mp4")
# 影片幀幅率
fps = cap.get(cv2.CAP_PROP_FPS)
F_Count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 影片影像尺寸
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 設定寫入格式（經過測試使用MP4V且輸出為.mov），幀幅率改為20.0
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter("video.mov", fourcc, 20.0, (w, h))

# 效果變數
# 2.(02-03m)Face Detector
detector = dlib.get_frontal_face_detector()

# MOG
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7)
timeF = 20

# KNN
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# ROI
RECT = ((600, 200), (1500, 850))
(left, top), (right, bottom) = RECT

# 取出 ROI 子畫面
def roiarea(frame):
    return frame[top:bottom, left:right]

# 將 ROI 區域貼回到原畫面
def replaceroi(frame, roi):
    frame[top:bottom, left:right] = roi
    return frame


# framediff
bb = None


while True and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 在設定的時間範圍為內的影像(frame)將呈現指定效果
    # 1.(00-02m)Laplacian
    if 0 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 2000:
        frame = cv2.Laplacian(frame, cv2.CV_64F)
        frame = cv2.convertScaleAbs(frame)
        cv2.putText(frame, "1.Laplacian", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 2.(02-03m)Face Detector
    if 2000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 3000:
        if ret:
            # 回傳人臉的位置, 分數, 子偵測器編號（idx）
            # 子偵測器的編號可以用來判斷人臉的方向；1為正面臉，調整為-0.5增加被辨識範圍
            face_rects, scores, idx = detector.run(frame, 0, -0.5)
            # 取出所有偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                text = f"{scores[i]:.2f}, ({idx[i]:0.0f})"
                # 以方框標示偵測的人臉
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                # 標示分數
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "2.Face Detector ", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 3.(03-04m)Flip 0 + COLORMAP_BONE
    if 3000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 4000:
        # 使影像上下左右顛倒
        frame = cv2.flip(frame, 0)
        # 套用COLORMAP系列
        frame = cv2.cvtColor(frame, cv2.COLORMAP_BONE)
        cv2.putText(frame, "3.Flip 0 + COLORMAP_BONE", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 4.(04-05m)MORPH_ELLIPSE
    if 4000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 5000:
        # cv2.getStructuringElement返回特定大小及形狀的結構元素(shape, ksize, anchor)
        me = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 10))
        frame = cv2.dilate(frame, me)
        cv2.putText(frame, " 4.MORPH_ELLIPSE", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 5.(05-06m)COLORMAP_COOL
    if 5000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 6000:
        # 套用COLORMAP系列
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_COOL)
        cv2.putText(frame, "5.COLORMAP_COOL", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 6.(06-07m)Sobel
    if 6000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 7000:
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        frame = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        cv2.putText(frame, " 6.Sobel", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 7.(07-08m)Flip -1 + BGR2RGB
    if 7000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 8000:
        # 使影像上下顛倒
        frame = cv2.flip(frame, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, "7.Flip -1 + BGR2RGB", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 8.(08-09m)Contour + HSV2BGR
    if 8000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 9000:
        # 邊緣(Edge)的線條頭尾相連形成封閉的區塊就是輪廓偵測(Contour)
        # 為了提高辨識率通常會先轉為黑白，物件必須是白色背景為黑色，於是需要邊緣運算的配合
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # 邊緣運算配合
        edged = cv2.Canny(blurred, 30, 150)
        # 獲取輪廓
        cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs = frame.copy()
        cv2.drawContours(objs, cnts, -1, (0, 255, 0), 2)
        # 主要目的為將二維(前處理的黑白影像)轉三維(三通道)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.putText(frame, "8.Contour + HSV2BGR", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 9.(09-11m)KNN + COLOR_GRAY2BGR
    if 9000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 11000:
        knn = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        frame = bs.apply(knn)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "9.KNN + COLORMAP_OCEAN", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 10.(11-12m)Canny + COLORMAP_PINK
    if 11000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 12000:
        frame = cv2.Canny(frame, 100, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "10.Canny + COLORMAP_PINK", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 11.(12-13m)Scharr
    if 12000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 13000:
        scharrx = cv2.Scharr(frame, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(frame, cv2.CV_64F, 0, 1)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        frame = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        cv2.putText(frame, "11.Scharr", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 12.(13-14m)MOG + COLORMAP_SUMMER
    if 13000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 14000:
        fgmask = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        frame = fgbg.apply(fgmask)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_SUMMER)
        cv2.putText(frame, "12.MOG + COLORMAP_SUMMER", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 13.(14-15m)ROI + COLORMAP_SPRING
    if 14000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 15000:
        # 取出子畫面
        roi = roiarea(frame)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_SUMMER)
        # roi = cv2.cvtColor(roi, cv2.COLORMAP_SPRING)
        # 將處理完的子畫面貼回到原本畫面中
        frame = replaceroi(frame, roi)
        # 在 ROI 範圍處畫個框
        cv2.rectangle(frame, RECT[0], RECT[1], (0, 0, 255), 2)
        cv2.putText(frame, "13.ROI + COLORMAP_SPRING", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 14.(15-17m)Frame Diff
    if 15000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 17000:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bb is None:
            bb = gray
            continue

        diff = cv2.absdiff(gray, bb)
        # binary 後, 只有黑和白 (0, 1)
        diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.erode(diff, None, iterations=2)
        diff = cv2.dilate(diff, None, iterations=2)

        cnts, hierarchy = cv2.findContours(
            diff,
            cv2.RETR_EXTERNAL,             # 若有重複, 只找外圈的
            cv2.CHAIN_APPROX_SIMPLE)  # 傳回特徵性座標, 如長 寬 高

        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "14.Frame Diff", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # 15.(17-18m)DOG
    if 17000 < cap.get(cv2.CAP_PROP_POS_MSEC) <= 18000:
        G0 = cv2.GaussianBlur(frame, (3, 3), 0)
        G1 = cv2.GaussianBlur(frame, (3, 3), 1, 1)
        # img_G1 = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = G0 + G1  # try img_G0 + img_G1
        cv2.putText(frame, " 15.DOG", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 2, cv2.LINE_AA)

    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類, 預設為正常可以不用設定True為上下顛倒)
    if 18000 < cap.get(cv2.CAP_PROP_POS_MSEC):

        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES)} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                    (100, 200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "openCV effect:", (1400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "1.Laplacian", (1400, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "2.Face Detector", (1400, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "3.Flip 0 + COLORMAP_BONE", (1400, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "4.MORPH_ELLIPSE", (1400, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "5.COLORMAP_COOL", (1400, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "6.Sobel", (1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "7.Flip -1 + BGR2RGB", (1400, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "8.Contour + HSV2BGR", (1400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "9.KNN + COLORMAP_OCEAN", (1400, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "10.Canny + COLORMAP_PINK", (1400, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "11.Scharr", (1400, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "12.MOG + COLORMAP_SUMMER", (1400, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "13.ROI + COLORMAP_SPRING", (1400, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "14.Frame Diff", (1400, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA),
        cv2.putText(frame, "15.DOG", (1400, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    out.write(frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)