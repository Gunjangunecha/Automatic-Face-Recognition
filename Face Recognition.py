#Importing necessary libraries
import cv2

#Loading the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Function to detect faces in the image provided
def detect_faces(image_path):
    print("Image Path:", image_path) 
    
    #Read the input image given
    img = cv2.imread(image_path)
    
    #Check if the image was successfully loaded
    if img is None:
        print("Error: Couldn't open the image file. Please check the file path.")
        return
    
    #Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    #Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    #Display the output image with detected faces
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Usage
if __name__ == "__main__":
    
    image_path = "D:\Controlone.ai Image Processing Engineer\input_image3.jpg" #Here you can change the path of the input image
    detect_faces(image_path)
