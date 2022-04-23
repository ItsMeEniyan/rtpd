import math
import cv2
import mediapipe as mp
import numpy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def cartesianDistance(x1, y1, z1, x2, y2, z2):
    D = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
    return D


def avgOfCurrPrevLandMarks(curr_landmarks, prev_landmarks):
    total_difference = 0

    for i in range(21):
        x1 = curr_landmarks[0].x
        y1 = curr_landmarks[0].y
        z1 = curr_landmarks[0].z
        x2 = prev_landmarks[0].x
        y2 = prev_landmarks[0].y
        z2 = prev_landmarks[0].z
        distance = cartesianDistance(x1, y1, z1, x2, y2, z2)

        total_difference += distance

    return total_difference / 21


def calculateCosineSimilarity(vectorSource, vectorCapture):
    cosine = np.dot(vectorSource, vectorCapture) / (norm(vectorSource) * norm(vectorCapture))
    return cosine


def normalizeIndex(i, l, r):
    return (i - l) / r


def mainVideoProcessingFunction():
    ## For webcam input:
    # cap = cv2.VideoCapture(0)
    ##For Video
    cap = cv2.VideoCapture("test.mp4")
    prevTime = 0
    loopItr = 0
    cartesianDistanceList = []

    with mp_hands.Hands(
            min_detection_confidence=0.5,  # Detection Sensitivity
            min_tracking_confidence=0.5) as hands:
        prev_landmarks = None
        prev_avg = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                if prev_landmarks:
                    avg = avgOfCurrPrevLandMarks(results.multi_hand_landmarks[0].landmark, prev_landmarks)
                    prev_avg = (avg + prev_avg) / 2
                prev_landmarks = results.multi_hand_landmarks[0].landmark
                if (loopItr < 263):
                    cartesianDistanceList.append(prev_avg * 1000)
                    loopItr += 1
                else:
                    break

                print("Mean Cartesian Coordinates Distance: ", prev_avg * 1000)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    ### FOR SIMILARITY INDEX

    vectorSource = [0, 0.5170588861891542, 20.740850401037243, 123.58376979333251, 173.78664675413776,
                    90.26517844836387, 153.83599483357307, 82.84581237064407, 57.5545378283527, 54.65551926497973,
                    34.06530284772292, 19.95770406086251, 12.57479478111175, 8.998224092124904, 120.41344495128118,
                    62.456746858906655, 60.99949897899223, 148.6566034209508, 76.33001459560116, 44.59005406556214,
                    28.66146190645213, 18.15575004572051, 113.24676569877884, 57.334988654286114,
                    124.05207956111686, 171.9189661566543, 195.36286030971277, 107.55724218589485,
                    62.808367250911104, 133.29658124425623, 167.6004704262279, 198.2613718852244,
                    102.06724899030586, 52.77450420357278, 27.43498461161053, 125.45847334658619, 75.76104065550797,
                    57.93069358588005, 45.09334110369789, 32.5176104491799, 30.219014102774942, 25.84039973120661,
                    126.20318667818286, 176.6247995151259, 89.14198997324779, 145.6749962399437, 74.65147910958056,
                    139.27091143331282, 182.90057401350117, 94.36110905283175, 158.08647919927287, 96.5210533495256,
                    55.09533827939676, 38.2878510849165, 121.6781153856598, 174.26012799719115, 90.27791803262525,
                    60.774133644267074, 55.4036979827455, 30.292050395872792, 17.024566122964558,
                    10.997797481866831, 9.169624646756597, 18.870595273963787, 126.8441634868737, 187.8889702164831,
                    97.89607485350662, 52.02552275377358, 130.1760426283134, 166.25024303160862, 204.32635716043873,
                    105.98270272846281, 55.38221489416972, 29.578661476642363, 123.20413665764987,
                    75.95084498314583, 54.81873166558325, 38.91789064222445, 137.68117443575693, 73.16526131185623,
                    157.05041662085443, 94.45680615235563, 54.540418014361855, 37.085919973577035,
                    38.40674992295563, 26.611333166140195, 26.893422200939394, 133.77380942189416,
                    69.19670363155788, 157.2946885491593, 97.3718969542608, 53.6064403091353, 31.719073226692757,
                    27.687999631096307, 126.86216404925968, 65.14265752526602, 34.114235565167014,
                    17.350303148911035, 128.13034988904684, 171.96326473615846, 88.6477650027327,
                    45.899490828869666, 23.630960058975678, 115.84672393492993, 175.32318060894474,
                    209.62980088172552, 112.12782728770834, 71.19215884671995, 39.25574251970481, 134.8807774196573,
                    74.93203152627518, 40.95665864281455, 27.50115790847754, 16.56922105867268, 133.61654243819066,
                    71.13237683969903, 139.34848832141645, 170.23684872570814, 203.38039413361403,
                    118.66933576881087, 65.15201962792686, 33.33144190277588, 128.40293408277574, 76.2603454406981,
                    49.77021616045843, 124.81455702663699, 162.31949084217717, 192.54794348633214,
                    209.28756810540838, 120.95715260844058, 75.07490759408407, 51.30643801783824, 38.7734559327092,
                    20.978348922944182, 27.743649831471732, 15.819883655707669, 9.63758857230967,
                    128.31203754944448, 181.30473821146248, 101.34645680086038, 62.63928958359159,
                    45.34824971869467, 27.589731869597024, 15.979231582998933, 129.28673798834873,
                    183.50359236907667, 103.68464620839272, 65.16685049210658, 129.95296942423872, 161.96176375261,
                    190.17017400468833, 98.06100372869321, 57.3440882739798, 140.49939883022944, 82.01957520184871,
                    51.62585559387195, 45.04940791751075, 48.5528171242644, 28.11836933611151, 16.945238720810465,
                    20.78421007559627, 126.80976378503524, 64.96093459576966, 163.6800701481368, 83.54596977660466,
                    55.89105734625546, 34.68778952258271, 26.179288869442285, 22.262554776880954,
                    20.324213654737175, 19.536415664264258, 20.463564204468806, 27.73862869061575,
                    18.311012899309738, 12.743405154896532, 18.302735285643802, 123.74199275430249,
                    67.97918783748773, 165.76658101201724, 87.13147852082041, 52.33464908360252, 33.8771944444952,
                    28.67744640397259, 26.693052623803844, 140.12035562197397, 186.23452104019466,
                    105.02907764094749, 58.10884606825601, 36.395848886161836, 18.464177112419293,
                    21.976047057782075, 134.33624736388606, 70.19819547294802, 37.51522137424527,
                    146.44747709323735, 80.97295961316183, 50.368509179169116, 38.30172454729885,
                    39.227619562891505, 36.44730641396872, 26.014700230473256, 125.85618635625931,
                    177.3099730652305, 104.41878928831429, 68.6028678047988, 44.71018481177526, 28.63346458037059,
                    29.34931668726201, 31.990012009456752, 33.92535862019676, 35.67840359558038, 21.22083847960572,
                    11.854394175288876, 7.147006555704866, 4.602370813888741, 3.667030036122128, 18.627512182684846,
                    20.77211893632714, 14.850512515577897, 15.55340477908609, 8.032516772643648, 13.735309446712328,
                    15.525454854427114, 10.493497820072673, 13.375693238278544, 116.15032320029533,
                    173.9700190084949, 95.06707820110115, 57.82747515048813, 39.65269175126104, 31.71158623661475,
                    37.17381919387924, 22.316578610889422, 12.21707770442406, 6.6773342330733145, 5.313917745203219,
                    4.491872574587673, 4.109619309368119, 3.2305408381073515, 3.432369722664479, 2.5131887949052785,
                    1.82778301243667, 1.8895847899417784, 1.9934053484311136, 1.767701966260906, 1.079357166074054,
                    1.4127724955846361, 3.502614090518789, 5.094091351386815, 11.638691500269628,
                    12.576812517936212, 8.671843556593434, 147.92977890327876, 215.26460446591273,
                    255.23798274876475, 352.4262237086474, 177.96033702009242, 89.64774641264403, 48.32681721771113,
                    26.627113560044265, 214.90923492201475, 116.19442675597267, 58.81608193542028]

    live_mean = numpy.mean(cartesianDistanceList)
    source_mean = numpy.mean(vectorSource)

    normalized_mean = normalizeIndex(live_mean, 0, source_mean)

    # plotGraph(cartesianDistanceList) if you want to plot the graph

    return normalized_mean * 100


def plotGraph(cartesianDistanceList):
    ### FOR PLOTTING - use it only for taking screenshots!

    plt.plot(cartesianDistanceList)

    plt.xlabel("frames")
    plt.ylabel("Mean Cartesian Coordinates Distance")
    plt.title("Hand - Tremor patterns from each frames")
    plt.show()
    return
    # tell matplotlib which yticks to plot
    # ax.set_yticks([0,1,2,3])

    # labelling the yticks according to your list
    # ax.set_yticklabels(['A','B','C','D'])


def doctorValidation(age, normalizedmean, normalizedmean30to50, normalizedmean50above):
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


def mainFunction():
    myNormalizedMean = mainVideoProcessingFunction()
    
    stringReturn = "Your Similarity Index value is: "+str(myNormalizedMean)+"%"

    print("Your Similarity Index value is: ", myNormalizedMean, "%", sep='')
    
    stringReturn += str(doctorValidation(64, myNormalizedMean, 35, 50))
    
    print(doctorValidation(64, myNormalizedMean, 35, 50))  # the input should be given by the doctor!

    return stringReturn