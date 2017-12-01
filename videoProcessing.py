from AverageFilter import *

cap = cv2.VideoCapture(0)

capture_fps= cap.get(cv2.CAP_PROP_FPS)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  start = time.time()

  ret, frame = cap.read()
  if ret == True:
    x,y,z=frame.shape
    #splitting R,G and B channel
    frame_r=frame[:,:,2]
    frame_g=frame[:,:,1]
    frame_b=frame[:,:,0]
    d_img_r=cuda.to_device(np.ascontiguousarray(frame_r))
    d_img_g=cuda.to_device(np.ascontiguousarray(frame_g))
    d_img_b=cuda.to_device(np.ascontiguousarray(frame_b))
    TPB=(kernel_width,kernel_width)
    BPG=(x,y)
  	#operating on R,G and B channel
    avg_filter[BPG,TPB](d_img_r,x-1,y-1)
    avg_filter[BPG,TPB](d_img_g,x-1,y-1)
    avg_filter[BPG,TPB](d_img_b,x-1,y-1)
    #combining three channels
    frame_smooth=np.dstack((d_img_g.copy_to_host(),d_img_g.copy_to_host(),d_img_r.copy_to_host()))


    cv2.imshow('Smoothed Image',frame_smooth)
    cv2.imshow('Original',frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 

  else: 
    break
  end=time.time()-start
  fps=60/end
  print(fps/capture_fps)


cap.release()
 

cv2.destroyAllWindows()
