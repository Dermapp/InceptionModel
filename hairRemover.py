class DigitalRazor:
  def removeHairs(image):
    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    #print( thresh2.shape )
    #plt.imshow(thresh2)
    #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)
    #plt.imshow(dst)
    #cv2.imwrite('final_image.jpg',dst,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return dst
