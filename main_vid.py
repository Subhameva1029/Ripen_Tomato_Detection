import numpy as np
import cv2 as cv

red_1 = np.array([0, 50, 180])

red_2 = np.array([10, 255, 255])

input_vid = cv.VideoCapture(0)

while (input_vid.isOpened()):
    ret, data_read = input_vid.read()
    if ret == True : 
        data_read = cv.flip(data_read, 1)
        data_read = cv.resize(data_read, (510, 510))
        image = cv.cvtColor(data_read, cv.COLOR_BGR2RGB)
        hsv_mode = cv.cvtColor(data_read, cv.COLOR_BGR2HSV)


        mask = cv.inRange(hsv_mode, red_1, red_2)
        red_mode = cv.bitwise_and(data_read, data_read, mask=mask)

        red_mode_rgb = cv.cvtColor(red_mode, cv.COLOR_BGR2RGB)
        complete_red = red_mode
        complete_red = cv.cvtColor(complete_red, cv.COLOR_BGR2RGB)
        face_tomato = cv.imread('_trial_model.jpeg')
        face_tomato = cv.cvtColor(face_tomato, cv.COLOR_BGR2RGB)
        result_final = cv.cvtColor(data_read, cv.COLOR_BGR2RGB)
        result_final = cv.resize(result_final, (510, 510))

        i = 0

        while True : 

            result = cv.matchTemplate(complete_red, face_tomato, cv.TM_SQDIFF)
            min_value, max_value, min_loc, max_loc = cv.minMaxLoc(result, None)
            height, width, channel = face_tomato.shape
            up_left = min_loc
            down_right = (up_left[0] + width , up_left[1] + height)
            cv.rectangle(complete_red, up_left, down_right, (5, 115, 5), 2, 8, 0)
            cv.rectangle(result_final, up_left, down_right, (5, 115, 5), 4, 8, 0)
            i = i+1
            if i == 6 :
                break


        result_final = cv.cvtColor(result_final, cv.COLOR_BGR2RGB)
        cv.imshow("Testing", result_final)
        if cv.waitKey(30) == ord('q'):
            break
    else :
        break

input_vid.release()

cv.destroyAllWindows()