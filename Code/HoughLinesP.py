import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    '''
    img: Origial image
    vertices: The coordinates of region of Interest
    
    '''
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    '''
    img: Original image
    lines: HoughLinesP lines
    
    '''
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

image = cv2.imread(r'F:\cityu_programming\Image processing project\Road\14.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
width, height = image.shape[1],image.shape[0]

'''
region_of_interest_vertices = [
    (0, 0),
    (0, height),
    (width, height),
    (width, 0)
]

'''

region_of_interest_vertices = [
    (0, 528),
    (246, 226),
    (310, 226),
    (width, 540)
]


gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

##########------Change of blur_image---------######

blur_image = cv2.GaussianBlur(gray_image, (5,5),0)


#blur_image = cv2.boxFilter(gray_image, -1, ksize = (3,3))

##########------Change of blur_image---------######


#canny_image = cv2.Canny(gray_image, 100, 200)

canny_image = cv2.Canny(blur_image, 100, 200)
cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32))

##########------Region of Interest---------######
Region_of_Interest = region_of_interest(gray_image,
                np.array([region_of_interest_vertices], np.int32))

plt.figure(1)
plt.imshow(Region_of_Interest, cmap = 'gray')
plt.axis('off')
plt.show()
##########------Region of Interest---------######


lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)
image_with_lines = drow_the_lines(image, lines)

plt.figure(2)
plt.imshow(image_with_lines)
plt.axis('off')
plt.show()

##########------Vedio Part---------######

def show_vedio():

    cap = cv2.VideoCapture(r"F:\cityu_programming\Image processing project\Vedio\Lane Detection Test Video 01.mp4")
    ret, frame = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'F:\cityu_programming\Image processing project\Vedio\output.mp4',fourcc, 20.0, (frame.shape[1],frame.shape[0]))
    
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur_gray = cv2.GaussianBlur(frame_gray, (5,5),0)
        
        canny_image = cv2.Canny(frame_blur_gray, 50, 200)
        vertices = [(270, 700), (580, 534), (670, 534),(913, 700)]
        
        cropped_image = region_of_interest(canny_image,
                    np.array([vertices], np.int32))
        lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi/180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
        frame_withlines = drow_the_lines(frame, lines)
        cv2.imshow("Lane Detection", frame_withlines)

        if ret == True:
            out.write(frame_withlines)

        if cv2.waitKey(1) == ord('z'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_vedio()
