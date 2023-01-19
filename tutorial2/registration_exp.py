import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import optimize


def print_fig(img1,img2):
    plt.figure(figsize = (10,10))
    
    plt.subplot(1,2,1)
    plt.imshow(img1,cmap='Greys_r')
    plt.title('image 1')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1,2,2)
    plt.imshow(img2,cmap='Greys_r')
    plt.title('image 2')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def print_hist(hist):
    plt.plot(hist, color='black')
    plt.bar(np.arange(hist.size),hist.flatten(),width=1)
    plt.show()

def print_graph(arr, name = 'ssd'):
    
    # ssd일때는 min으로, mi일때는 max로 수정해야함
    if name == 'ssd':
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), subplot_kw={"projection":"3d"})
        x = np.arange(-30,30)
        y = np.arange(-30,30)
        X,Y = np.meshgrid(x,y)
        Z = arr
        x_index, y_index  = np.unravel_index(np.argmin(Z),Z.shape)
        label = f'({x[x_index]},{y[y_index]})'
        ax.plot_surface(X,Y,Z, cmap='plasma')
        ax.view_init(elev=20., azim=210)
        ax.scatter(y[y_index],x[x_index],min(Z.ravel()))
        print(max(Z.ravel()))
        ax.text(y[y_index],x[x_index]-5,min(Z.ravel()),label,ha='left')
        plt.show()
    else:
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), subplot_kw={"projection":"3d"})
        x = np.arange(-30,30)
        y = np.arange(-30,30)
        X,Y = np.meshgrid(x,y)
        Z = arr
        x_index, y_index  = np.unravel_index(np.argmax(Z),Z.shape)
        label = f'({x[x_index]},{y[y_index]})'
        ax.plot_surface(X,Y,Z, cmap='plasma_r')
        ax.view_init(elev=20., azim=210)
        ax.scatter(y[y_index],x[x_index],max(Z.ravel()))
        ax.text(y[y_index],x[x_index]-5,max(Z.ravel()),label,ha='left')
        plt.show()


def cal_ssd(image1, image2):
    ssd = np.sum((image1 - image2)**2)
    return ssd

def cal_hist(img):
    # histogram 계산
    hist, _ = np.histogram(img, bins=256)
    return hist

def cal_jointHist(img1,img2):
    # joint histogram계산
    hist1 = cal_hist(img1)
    hist2 = cal_hist(img2)
    jointHist, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins = 256)
    jointHist = jointHist / jointHist.sum()
    return jointHist, hist1, hist2

def cal_Entropy(Hist):
    # histgram으로 entropy 계산
    Hist_ = Hist / Hist.sum()
    Hist_ = Hist_[np.nonzero(Hist_)[0]]
    H = -np.sum(Hist_ * np.log2(Hist_))
    return H

def cal_MI(image1, image2):
    # MI 계산
    jointHist, hist1, hist2 = cal_jointHist(image1, image2)
    hist1 = hist1[hist1 > 0]
    hist2 = hist2[hist2 > 0]
    jointHist = jointHist[jointHist > 0]
    H_AB = cal_Entropy(jointHist)
    H_A = cal_Entropy(hist1)
    H_B = cal_Entropy(hist2)
    return H_A + H_B - H_AB

def ssd(params,fixed_image, moving_image):
   
    tx, ty, tr = params
    (h, w) = moving_image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    trsl_m = np.float32([[1, 0, tx],[0, 1, ty]])
    img1_shift = cv2.warpAffine(moving_image , trsl_m, (moving_image.shape[1],moving_image.shape[0]))
    rot_m = cv2.getRotationMatrix2D((cX, cY), tr, 1.0)
    rot_img = cv2.warpAffine(img1_shift, rot_m, (w, h))
    
    return cal_ssd(fixed_image,rot_img)


def MI(params, fixed_image, moving_image):
    
    tx, ty, tr = params
    (h, w) = moving_image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    trsl_m = np.float32([[1, 0, tx],[0, 1, ty]])
    img1_shift = cv2.warpAffine(moving_image , trsl_m, (moving_image.shape[1],moving_image.shape[0]))
    rot_m = cv2.getRotationMatrix2D((cX, cY), tr, 1.0)
    rot_img = cv2.warpAffine(img1_shift, rot_m, (w, h))
    
    return -cal_MI(fixed_image,rot_img)

    
def img_read():
    img1 = cv2.imread('image1.tif',0)
    img2 = cv2.imread('image2.tif',0)
    img1 = cv2.copyMakeBorder(img1, 100,100,100,100, cv2.BORDER_CONSTANT, value=[0,0]) #위,아래,왼,오 0으로 패딩
    # img2 = cv2.copyMakeBorder(img2, 100,100,100,100, cv2.BORDER_CONSTANT, value=[0,0]) #위,아래,왼,오 0으로 패딩
    print_fig(img1,img2)
    return img1, img2

def optm_param(x_range, y_range, similarity):
    if len(similarity.shape) == 2:
        min_index = np.argmin(similarity)
        x_index, y_index = np.unravel_index(min_index,similarity.shape)
        x_ = x_range[x_index]
        y_ = y_range[y_index]
        return x_, y_
    else :
        min_index = np.argmin(similarity)
        x_index, y_index, r_index = np.unravel_index(min_index,similarity.shape)
        x_ = x_range[x_index]
        y_ = y_range[y_index]
        r_ = np.arange(360)[r_index]
        return x_, y_,r_

def image_registration_ssd(img1,img2,x_range,y_range):
    ssd = np.zeros((len(y_range), len(x_range)))
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # image translation & calc ssd
            trsl_m = np.float32([[1, 0, x],[0, 1, y]]) 
            img2_shift = cv2.warpAffine(img2, trsl_m, (img2.shape[1],img2.shape[0]))
            ssd[i,j] = cal_ssd(img1,img2_shift)
    return ssd

def image_registration_mi(img1,img2,x_range,y_range):
    MI = np.zeros((len(y_range),len(x_range)))
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            trsl_m = np.float32([[1, 0, x],[0, 1, y]]) 
            img2_shift = cv2.warpAffine(img2, trsl_m, (img2.shape[1],img2.shape[0]))
            MI[i,j] = -cal_MI(img1,img2_shift)
    return MI

def image_registration(img1,img2,x_range,y_range):
    
    ssd = np.zeros((len(y_range), len(x_range), 60))
    MI = np.zeros((len(y_range),len(x_range),60))
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, r in enumerate(np.arange(60)):
                (h, w) = img2.shape[:2]
                (cX, cY) = (w / 2, h / 2)
                trsl_m = np.float32([[1, 0, x],[0, 1, y]])
                img2_shift = cv2.warpAffine(img2 , trsl_m, (img2.shape[1],img2.shape[0]))
                rot_m = cv2.getRotationMatrix2D((cX, cY), r, 1.0)
                rot_img = cv2.warpAffine(img2_shift, rot_m, (w, h))
                ssd[i,j,k] = cal_ssd(img1,rot_img)
                MI[i,j,k] = -cal_MI(img1,rot_img)
                if i%5==0 and j==0 and k== 0:
                    print(i)
    return ssd , MI



def interpolation(img, size):
    nearest_neighbor_image  = cv2.resize(img, dsize= size,interpolation=cv2.INTER_NEAREST)
    bilinear_image = cv2.resize(img, dsize= size,interpolation=cv2.INTER_LINEAR)
    bicubic_image = cv2.resize(img, dsize= size, interpolation=cv2.INTER_CUBIC)
    return nearest_neighbor_image, bilinear_image, bicubic_image


def Q1(img1, img2, x_range, y_range):
    ssd= image_registration_ssd(img1,img2,x_range,y_range)
    print(min(ssd.ravel()))       
    x_, y_ = optm_param(x_range,y_range,ssd)
    print(x_,y_)
    
    print_fig(img1,cv2.warpAffine(img2, np.float32([[1, 0, x_],[0, 1, y_]]), (img2.shape[1],img2.shape[0])))
    print_graph(ssd)

def Q2(img1, img2, x_range, y_range):
    MI = image_registration_mi(img1,img2,x_range,y_range)
    x_, y_ = optm_param(x_range,y_range,MI)
    print(x_,y_)
    print(max(-MI.ravel()))   
    print_fig(img1,cv2.warpAffine(img2, np.float32([[1, 0, x_],[0, 1, y_]]), (img2.shape[1],img2.shape[0])))
    print_graph(-MI, name = 'mi')
    
def Q3(img1, x_range, y_range):
    (h, w) = img1.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    print(cX,cY)
    rot_m = cv2.getRotationMatrix2D((cX, cY), -45, 1.0)
    rot_img = cv2.warpAffine(img1, rot_m, (w, h))
    trsl_m = np.float32([[1, 0, -10],[0, 1, 8]]) 
    img1_trsf = cv2.warpAffine(rot_img, trsl_m, (img1.shape[1],img1.shape[0]))
    print(rot_img.shape)
    print_fig(img1,img1_trsf)
    
    ssd, MI = image_registration(img1, img1_trsf, x_range, y_range)
    x_ssd, y_ssd,r_ssd = optm_param(x_range,y_range,ssd)
    x_mi, y_mi, r_mi = optm_param(x_range,y_range,MI)
    print("ssd 기준 : ", x_ssd, y_ssd, r_ssd, "\n MI 기준 : ", x_mi, y_mi, r_mi)
    print_graph(ssd[:,:,45])
    print_graph(-MI[:,:,45], name = 'mi')
    print(min(ssd.ravel()))
    print(max(-MI.ravel()))
    
    rot_m_ssd = cv2.getRotationMatrix2D((cX, cY), r_ssd, 1.0)
    
    shf_img_ssd = cv2.warpAffine(img1_trsf, np.float32([[1, 0, x_ssd],[0, 1, y_ssd]]), 
                   (img1_trsf.shape[1],img1_trsf.shape[0]))
    rot_img_ssd = cv2.warpAffine(shf_img_ssd, rot_m_ssd, (w, h))
    
    
    rot_m_mi = cv2.getRotationMatrix2D((cX, cY), r_mi, 1.0)
    shf_img_mi = cv2.warpAffine(img1_trsf, np.float32([[1, 0, x_mi],[0, 1, y_mi]]), 
                   (img1_trsf.shape[1],img1_trsf.shape[0]))
    rot_img_mi = cv2.warpAffine(shf_img_mi, rot_m_mi, (w, h))
    
    
    # 결과 이미지 출력
    print_fig(rot_img_ssd,rot_img_mi)
    
    plt.figure(figsize=(20,15))
    plt.subplot(1,3,1)
    plt.imshow(img1,cmap='Greys_r')
    plt.title("original")
    plt.subplot(1,3,2)
    plt.imshow(img1_trsf,cmap='Greys_r')
    plt.title("distortion")
    plt.subplot(1,3,3)
    plt.imshow(rot_img_mi,cmap='Greys_r')
    plt.title("result_MI")
    plt.show()
    
    plt.figure(figsize=(20,15))
    plt.subplot(1,3,1)
    plt.imshow(img1,cmap='Greys_r')
    plt.title("original")
    plt.subplot(1,3,2)
    plt.imshow(img1_trsf,cmap='Greys_r')
    plt.title("distortion")
    plt.subplot(1,3,3)
    plt.imshow(rot_img_ssd,cmap='Greys_r')
    plt.title("result_SSD")
    plt.show()
    
    

def Q4(img1):
    nearest_1, bilinear_1, bicubic_1 = interpolation(img1[64:193,64:193], (256,256)) # 128,128 
    nearest_2, bilinear_2, bicubic_2 = interpolation(img1[64:193,64:193], (64,64)) # 128,128 
    plt.figure(figsize=(20,15))
    for i, name in enumerate(('nearest_1', 'bilinear_1', 'bicubic_1', 'nearest_2', 'bilinear_2', 'bicubic_2')):
          plt.subplot(2,3,i+1)
          plt.imshow(eval(name),cmap='Greys_r')
          plt.title(name)
          plt.xticks([])
          plt.yticks([])
    plt.show()

def Q5(img1, x_range, y_range):
    (h, w) = img1.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    # transform image (rotation :  -45, x : -10, y :  8)
    rot_m = cv2.getRotationMatrix2D((cX, cY), -45, 1.0)
    rot_img = cv2.warpAffine(img1, rot_m, (w, h))
    trsl_m = np.float32([[1, 0, -10],[0, 1, 8]]) 
    img1_trsf = cv2.warpAffine(rot_img, trsl_m, (img1.shape[1],img1.shape[0]))
    print_fig(img1,img1_trsf)
    
    
    x0 = np.array([-30,30,10]) # 초기값 설정
    result_ssd = optimize.fmin(ssd,x0,args=(img1,img1_trsf)) # fmin 이용하여 ssd를 최소화
    result_mi = optimize.fmin(MI,x0,args=(img1,img1_trsf)) # fmin 이용하여 MI를 최소화
    x_ssd, y_ssd  = result_ssd[:2]
    r_ssd = result_ssd[2]
    
    # Matirx 구하기
    M_ssd=  np.float32([[np.cos(r_ssd), -np.sin(r_ssd), x_ssd],
                        [np.sin(r_ssd), np.cos(r_ssd), y_ssd],
                        [0, 0, 1]])
    
    print("\n <ssd> \n translation : ", x_ssd , y_ssd, "\n rotation    : ",r_ssd)
    print("\n <SSD Matrix> \n", M_ssd)

    x_mi, y_mi  = result_mi[:2]
    r_mi = result_mi[2]
    
    
    M_mi =  np.float32([[np.cos(r_mi), -np.sin(r_mi), x_mi],
                        [np.sin(r_mi), np.cos(r_mi), y_mi],
                        [0, 0, 1]])
    
    print("\n <MI> \n translation : ", x_mi , y_mi,"\n rotation    : ", r_mi)
    print("\n <MI Matrix> \n", M_mi)
    

    
    
    
    

def main():
    img1, img2 = img_read()
    x_range = np.arange(-30,30, 1)
    y_range = np.arange(-30,30, 1)

    # Q1(img1, img2, x_range, y_range)
    # Q2(img1, img2, x_range, y_range)
    Q3(img1, x_range, y_range)
    # Q4(img1)
    # Q5(img1, x_range, y_range)
    
    
    return 0

main()
