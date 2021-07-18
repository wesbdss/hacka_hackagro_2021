import cv2
import numpy as np
from datetime import datetime

vid = cv2.VideoCapture('./videos/videoex.mp4')
quadro = 0
timer = datetime.now().time()
while(True):
    quadro += 1
    grabbed, frame = vid.read()

    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_LINEAR)

    if not grabbed:
        break
    # separar as layers
    try:
        base = np.array(frame.copy())
        image = np.array(frame.copy())
    except Exception as ex:
        continue

    ###############################################
    ###############################################
    ###############################################
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_correct = np.array(255 * (image / 255) ** 1.2 , dtype='uint8')

    # thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
    # thresh = cv2.bitwise_not(thresh)

    # kernel = np.ones((15,15), np.uint8)
    # img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # img_erode = cv2.erode(img_dilation,kernel, iterations=1)

    # img_erode = cv2.medianBlur(img_erode, 7)

    # cv2.imshow('frame', img_erode)

    ###############################################
    ###############################################
    ###############################################
    # image = np.absolute((2*(base[:,:,2] - base[:,:,1] - base[:,:,0]))/(2*(base[:,:,2] + base[:,:,1] + base[:,:,0])))

#     B = base[:,:,0].astype(float)
#     # G
#     G = base[:,:,1].astype(float)
#     G = G *1.3
#     # R
#     R = base[:,:,2].astype(float)
#     R = R *0.2
#     # try:
#     #     a = np.add(base[:,:,2], base[:,:,1], base[:,:,0])
#     #     b = np.subtract(base[:,:,2], base[:,:,1], base[:,:,0])
#     #     gli = np.divide(np.multiply(2,b),np.multiply(2,a))
#     # except Exception:
#     #     continue

#     # another equation

#     # cv2.imshow('frame', gli)

# #
# #
# #
#     # NDVI visivel
#     try:
#         a = np.subtract(G, R)
#         b = np.add(G, R)
#         image = np.divide(a,b)
#     except Exception:
#         continue

#     # normalize1 = np.divide(image,np.linalg.norm(image))

#     # identifica a planta NDVI alto
#     # mask = cv2.inRange(image, 0.71,)
#     mask = cv2.inRange(image, 0.72,1)
#     mask = cv2.GaussianBlur(mask, (9,9),0)
#     mask = cv2.Canny(mask, 200, 500)

#     # identifica o dainha
#     # mask2 = cv2.inRange(image, 0.69, .71)

#     # # Chão mais a daninha
#     # mask3 = cv2.inRange(image,0,0.69)

#     tirar_borda = cv2.bitwise_and(image,image, mask=mask)
#     # tirar_chao = cv2.bitwise_not(tirar_borda,tirar_borda, mask=mask3)


#     # res = cv2.bitwise_and(tirar_borda)

#     ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(mask,connectivity=4)
#     label_hue1 = np.uint8(179 * labels1 / np.max(labels1))
#     blank_ch1 = 255 * np.ones_like(label_hue1)
#     blank1_ch1 = 205 * np.ones_like(label_hue1)
#     labeled_img1 = cv2.merge([label_hue1, blank_ch1, blank1_ch1])
#     labeled_img1 = cv2.cvtColor(labeled_img1, cv2.COLOR_HSV2BGR)
#     labeled_img1[label_hue1 == 0] = 0

#     cv2.imshow('frame', labeled_img1)

    ###############################################
    #####################ENTREGÁVEL até o espaço##########################
    ###############################################

    img_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    lower_green = np.array([13, 0, 0])
    upper_green = np.array([100, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_green, upper_green)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask1, connectivity=4)

    mask = np.zeros(mask1.shape, dtype="uint8")

    for i in range(1, ret):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keepWidth = w > 5 and w < 50
        # keepHeight = h > 45 and h < 65
        keepArea = area > 70 and area < 200
        # if all((keepWidth, keepHeight, keepArea)):
        if all((keepArea,)):
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

    ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(
        mask, connectivity=4)
    label_hue1 = np.uint8(179 * labels1 / np.max(labels1))
    blank_ch1 = 255 * np.ones_like(label_hue1)
    blank1_ch1 = 205 * np.ones_like(label_hue1)
    labeled_img1 = cv2.merge([label_hue1, blank_ch1, blank1_ch1])
    labeled_img1 = cv2.cvtColor(labeled_img1, cv2.COLOR_HSV2BGR)
    labeled_img1[label_hue1 == 0] = 0

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    blank1_ch = 205 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank1_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    # piscar linha ao passar
    # for (x, y) in centroids1:
    #     print(x,y)
    #     if np.absolute(x-300) < 30:
    #         cv2.line(base, (0, 300), (400, 300), (0, 0, 255), 2)
    #         timer = datetime.now().time()
    #         timer[1] = timer[1]+3
    #         print("Passou")
    #         break
    #     else:
    #         cv2.line(base, (0, 300), (400, 300), (255, 0, 0), 2)
    #         break
    cv2.line(base, (0, 300), (400, 300), (0, 0, 255), 1)
    flag = 0
    sensor = 0
    for (x, y) in centroids1:
        if y <= 300:
            flag +=1
        if y <=350 and y >=270:
            sensor = 1
            break
        
    if sensor:
        cv2.line(base, (0, 300), (400, 300), (0, 0, 255), 1)
    else:
        cv2.line(base, (0, 300), (400, 300), (255, 0, 0), 2)
    if flag <=4:
        cv2.putText(base, "Desligado: "+str(flag), (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255),2)
    else:
        cv2.putText(base, "Ligado: "+str(flag), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50),2)

            
    # if flag== 1:
    #     cv2.putText(base, "Desligado", (30, 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0))
            

    # sizes = stats[1:, -1]
    # min_size= 700
    # max_size = 1500
    # try:
    #     for i,c in enumerate(centroids):
    #         # print(sizes[i],sizes[i] >= min_size)
    #         if sizes[i] >= min_size and sizes[i]<=max_size :
    #             try:
    #                 cv2.circle(base, (int(c[0]), int(c[1])),10,(255, 0,0),5)
    #             except Exception as ex:
    #                 print("erro")
    # except Exception:
    #     pass
    cv2.imshow("Processada", labeled_img)
    cv2.imshow("Original", base)
    cv2.imshow("Somente Daninhas", labeled_img1)

    # cv2.imwrite('allframes/'+str(quadro)+'.png', labeled_img)
    # print("frame "+str(quadro)+" salvo")
    ###############################################
    ###############################################
    ###############################################

    # GLI = (2 x G-R-B) / (2 x G+R+B)

    # mascaras de extração de cores (só achar o ponto verde)

    # image = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2HSV)

    # upper_green = np.array([131,51,100])
    # lower_green = np.array([156,97,100])

    # mask = cv2.inRange(image, (30, 0, 0),(91, 255,255))

    # res = cv2.bitwise_and(frame,frame, mask= mask)

    # cv2.imshow('frame', image)
    # cv2.imshow('frame2', res)

    ###############################################
    ###############################################
    ###############################################
    # cv2.imshow('frame', mask1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
