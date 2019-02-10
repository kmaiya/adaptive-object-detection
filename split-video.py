import cv2

########################################################
# Purpose of This Script:                              #
#   - To decode a video stream into image frames       #
#                                                      #
# Global Variables:                                    #  
#   VIDEO_PATH:                                        #
#       - Path to the video to stream                  #
#   VIDEO_PREFIX:                                      # 
#       - Prefix of the video stream. For EX: if the   #
#       video is called 'bunnyhopping.mp4' the prefix  # 
#       should be bunnyhopping. This prefix is used to #
#       save the image frames                          #  
#                                                      #     
#   SAVE_DIR:                                          #     
#       - Directory to store the extracted frames into #
########################################################

VIDEO_PATH      =   ''
VIDEO_PREFIX    =   ''
SAVE_DIR        =   ''

#Directory to save the images to
def main():

    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(SAVE_DIR+VIDEO_PREFIX+'_'+str(count).zfill(6)+'_.jpg', image) #save image as JPEG
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
