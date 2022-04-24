import math
import cv2
import mediapipe as mp
import time

import numpy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def cartesianDistance(x1, y1, z1, x2, y2, z2):
    D = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
    return D


def avgOfCurrPrevLandMarks(curr_landmarks, prev_landmarks):
    total_difference = 0
    needed_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]

    for i in range(33):
        if i in needed_indices:
            x1 = curr_landmarks[i].x
            y1 = curr_landmarks[i].y
            z1 = curr_landmarks[i].z
            x2 = prev_landmarks[i].x
            y2 = prev_landmarks[i].y
            z2 = prev_landmarks[i].z
            distance = cartesianDistance(x1, y1, z1, x2, y2, z2)

            total_difference += distance

    return total_difference / 10


def calculateCosineSimilarity(vectorSource, vectorCapture):
    cosine = np.dot(vectorSource, vectorCapture) / (norm(vectorSource) * norm(vectorCapture))
    return cosine

def normalizeIndex(i, l, r):
    return (i - l) / r


def mainVideoProcessingFunction():
    # For webcam input:
    # cap = cv2.VideoCapture(0)
    # For Video input:
    cap = cv2.VideoCapture('test.mp4')
    prevTime = 0
    cartesianDistanceList = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        prev_landmarks = None
        prev_avg = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            ### LOGIC HERE ###

            if results.pose_landmarks:
                # print(results.pose_landmarks.landmark[11].x)
                # break
                if prev_landmarks:
                    avg = avgOfCurrPrevLandMarks(results.pose_landmarks.landmark,
                                                 prev_landmarks)  ## currlandmarkslist, prevlandmarkslist
                    prev_avg = (avg + prev_avg) / 2
                prev_landmarks = results.pose_landmarks.landmark  # find the object

                cartesianDistanceList.append(prev_avg * 1000)
                print("Mean Cartesian Coordinates Distance: ", prev_avg * 1000)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            cv2.imshow('BlazePose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    ##declare vectorSource list
    vectorSource = [0, 32.98712993092826, 19.65252514732791, 62.93767489180579, 51.42545664361198, 56.11776285388532,
                    19.330391411145662, 45.179591323933295, 25.073178690096988, 50.166008366785775, 16.37058284846399,
                    19.035251243861637, 23.29280183648625, 27.94776494690276, 55.65154253260752, 25.2382435358113,
                    40.247736412445605, 60.10167128867408, 64.74240571996069, 24.143810623990205, 25.418796224790228,
                    61.335398536331496, 24.15362799413583, 39.59077466317274, 54.59752798564281, 41.787048613998635,
                    43.81284411547787, 31.349278216586256, 30.4303911139672, 21.62707455040308, 64.00996480277026,
                    62.89363149995551, 20.29462108021278, 53.58964550222365, 26.86283921108579, 53.05878339466991,
                    38.20593912910974, 51.30144134881026, 64.66856642263966, 61.24277955314077, 43.69753192662307,
                    35.428641984826314, 43.98698502396458, 47.35662207169898, 30.61346030668624, 37.24963418809442,
                    53.62866690612485, 24.169708937758337, 49.87033455557596, 28.90685723984698, 23.848698085213627,
                    65.33216042659956, 55.496005321480936, 33.44105912410474, 66.08163273243647, 54.30667168583063,
                    38.102236887513754, 50.853649833469554, 57.77163572168105, 45.13216516679028, 42.638082909774184,
                    17.07109623172246, 47.20412375832348, 35.10774825078491, 66.4552986302941, 36.422820696878986,
                    66.62561507524732, 53.82194854906558, 18.3131146779576, 17.183072774152702, 22.565742750924773,
                    46.435956789964756, 53.820905875984145, 21.064888751685864, 24.601579253783985, 19.507633141431103,
                    21.137920000158644, 35.18603641613298, 55.623440077817484, 33.49930543900844, 30.626516572048523,
                    28.2855046812865, 27.404739148058873, 63.31207232950318, 36.719760999465805, 17.13936275895897,
                    60.464211029534994, 23.72526210862198, 64.65000844725822, 48.532071974794164, 46.33103005592717,
                    36.57426908732867, 20.61572599786909, 63.76836262779663, 63.29414272938296, 54.74030098628662,
                    22.576720329157578, 52.11928188146161, 65.47850845382484, 64.96500390310912, 66.59836497140735,
                    58.256722753146924, 17.11134654077819, 19.095632138447357, 19.759659703349513, 23.645135318520058,
                    57.02723166742887, 65.46613394831577, 44.20409075881197, 24.82700883353374, 42.7514244654715,
                    42.334555175331026, 31.77999155326995, 63.37238962790195, 56.1592864695319, 52.400504620203584,
                    37.46315704567194, 24.773567244331364, 54.008188107071184, 19.06531470335522, 64.75190295323242,
                    45.8617857990962, 60.30759973317029, 38.95167104070832, 57.34753142676351, 59.87406976837871,
                    44.975159179949, 55.40507803768958, 52.39907663980743, 63.12331445336707, 30.047032904477852,
                    40.0815574414698, 40.45768743600384, 36.54794471924829, 36.65639780255027, 28.00385058059738,
                    34.987155482135314, 51.972782509912655, 40.47636901971427, 65.8887204347947, 65.13775185350232,
                    56.65940328608733, 33.18444646369934, 25.186747033567045, 59.318238925478056, 50.72737514644127,
                    32.84667108941728, 23.204555961934343, 48.48854457690271, 22.37015786533932, 24.03190528749533,
                    18.88702396448544, 50.4882524112706, 36.79206808649242, 26.384489100568317, 17.749729553650507,
                    17.45008024494477, 20.769114468185307, 62.2451458217256, 29.73637970025373, 34.26239437710112,
                    56.632006081834426, 32.759274600411544, 48.359436292945496, 37.43490970166575, 41.38411809150118,
                    22.119233980991584, 63.314664265172716, 29.67707494409813, 51.5933435199604, 45.432061444728994,
                    26.408724153484293, 43.088844951943, 26.821171816616804, 32.23307731040445, 40.4536475418394,
                    17.666535528167422, 62.2185182964517, 32.00496949227654, 41.74534303524147, 57.003927569042176,
                    51.50241646760976, 20.927448205219825, 66.07111614538465, 33.012753390981544, 54.4883647444734,
                    61.78950410296688, 20.026082570235893, 64.16146933967505, 42.16759169496899, 48.9237805810209,
                    22.70949740951554, 60.30718993612961, 33.29610150443537, 50.16403010510448, 40.04900898066674,
                    28.083073078551397, 39.407927335882746, 24.134331386940627, 32.00350072665727, 30.157987486616342,
                    64.01946550008182, 45.298219855282824, 41.34576999130584, 48.28278004512353, 22.49254903396254,
                    61.055083676006575, 31.106754036099687, 44.81634840940792, 58.50525285683832, 35.60499035964374,
                    49.5101342243062, 32.033588200101505, 18.0269139422596, 47.2607760545004, 25.741599692798303,
                    43.75321760788794, 53.12172412720631, 28.623215533005464, 32.14755086025477, 65.56375705630339,
                    47.864514777738094, 61.23678380823483, 36.109670532823685, 20.474410781361026, 52.71492977500768,
                    61.03381551519091, 40.23611483038128, 23.06155488705692, 43.153280392571816, 31.06755017715561,
                    62.526807484494725, 26.31940861354172, 18.577380962887474, 32.521872209879994, 20.217497927914344,
                    60.92192473064832, 53.45629234703565, 17.8418604260137, 57.436821144227274, 52.15715682191661,
                    61.82659894198567, 56.17376367012724, 49.972495008632926, 65.68510703778774, 64.75638674767399,
                    60.45727527534565, 57.21416762674505, 55.353560146409464, 45.629645556272365, 32.58876703706987,
                    52.16400223059017, 34.424163251222396, 28.07070221197148, 53.69961481029699, 26.064830265749556]

    live_mean = numpy.mean(cartesianDistanceList)
    source_mean = numpy.mean(vectorSource)

    normalized_mean = normalizeIndex(live_mean, 0, source_mean)

    # plotGraph(cartesianDistanceList) if you want to plot the graph

    return normalized_mean * 100

def plotGraph(cartesianDistanceList):
        ### FOR PLOTTING

        plt.plot(cartesianDistanceList)

        plt.xlabel("frames")
        plt.ylabel("Mean Cartesian Coordinates Distance")
        plt.title("Body - Tremor patterns from each frames")
        plt.show()

        return


def doctorValidation(age, normalizedmean, normalizedmean30to50, normalizedmean50above):

    # change the hardcoded values to your need later!

    if age > 30 and age < 50:
        if normalizedmean > normalizedmean30to50:  # 35
            return "You might have PDD"
        else:
            return "Normal Patient"
    elif age >= 50:
        if normalizedmean >= normalizedmean50above:  # 50
            return "You migh have PDD"
        elif normalizedmean > normalizedmean30to50 and normalizedmean < normalizedmean50above:  # 35
            return "You have to visit your nearest Nuerological Surgeon. You show symptoms of starting stage of PDD"
        else:
            return "Normal Patient"


def mainFunction(age):

    myNormalizedMean = mainVideoProcessingFunction()
    myNormalizedMean = round(myNormalizedMean, 2)

    stringReturn = "Your Similarity Index value is: " + str(myNormalizedMean) + "%"

    print("Your Similarity Index value is: ", myNormalizedMean, "%", sep='')

    stringReturn += str(doctorValidation(age, myNormalizedMean, 35, 50))

    print(doctorValidation(age, myNormalizedMean, 35, 50))  # the input should be given by the doctor!

    return stringReturn

