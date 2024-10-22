
# Importation des bibliothèques

from ultralytics import YOLO
import numpy as np
import cvzone
import cv2
import math
import vonage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import configparser
import logging
import time
import numpy as np
from email.mime.application import MIMEApplication







# la configuration de fichier de journalisation app.log
logging.basicConfig(
    level=logging.INFO, 
    filename='app.log', 
    filemode='a',       
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



#  lire les valeurs du fichier de configuration config.config
def read_config_file(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    general_config = config['General']
    stream = general_config.get('stream')

    nexmo_config = config['Nexmo']
    nexmo_key = nexmo_config.get('key')
    nexmo_secret = nexmo_config.get('secret')
    nexmo_from = nexmo_config.get('from')
    nexmo_to = nexmo_config.get('to')

    email_config = config['Email']
    email_address = email_config.get('email_address')
    email_password = email_config.get('email_password')
    email_addressre = email_config.get('email_addressre')

    return nexmo_key, nexmo_secret, email_address, email_password, email_addressre, nexmo_from, nexmo_to,stream

# enregistrer les valeurs dans le fichier de configuration config.config
def save_config_file(config_file, nexmo_key, nexmo_secret, email_address, email_password, email_addressre, nexmo_from, nexmo_to,stream):

    config = configparser.ConfigParser()
    config['General'] = {
        'stream':str(stream)
    }

    config['Nexmo'] = {
        'key': nexmo_key,
        'secret': nexmo_secret,
        'from' : nexmo_from,
        'to': nexmo_to
    }

    config['Email'] = {
        'email_address': email_address,
        'email_password': email_password,
        'email_addressre': email_addressre
    }

    with open(config_file, 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    config_file = 'config.config'

    # Lire les valeurs de configuration à partir du fichier
    nexmo_key, nexmo_secret, email_address, email_password, email_addressre, nexmo_from, nexmo_to,stream = read_config_file(config_file)
    
    # envoie un message SMS en utilisant l'API Vonage
    def send_sms(confidence , bol):

        client = vonage.Client(key= nexmo_key, secret= nexmo_secret)
        sms = vonage.Sms(client)

        if bol == True :
            responseData = sms.send_message(
                {
                    "from": nexmo_from,
                    "to": nexmo_to,
                    "text": f"Une detection avec une confiance superieure à {round(confidence, 2)} % a ete effectuee.\n",
                }
            )

            if responseData["messages"][0]["status"] == "0":

                print("Message sent successfully.")
                logging.info("Message SMS envoyé avec succès.")

            else:

                print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
                logging.error(f"Message failed with error: {responseData['messages'][0]['error-text']}")
        
        if bol == False :

            responseData = sms.send_message(
                {
                    "from": nexmo_from,
                    "to": nexmo_to,
                    "text": f"La dérniere detection probablement erroné \n",
                }
            )

            if responseData["messages"][0]["status"] == "0":

                print("Message sent successfully.")
                logging.info("Message SMS envoyé avec succès.")

            else:

                print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
                logging.error(f"Message failed with error: {responseData['messages'][0]['error-text']}")



    # envoie un e-mail avec une image jointe
    def send_email(frame , confidence , video_path):

        subject = f'Alerte - Confiance supérieure à{round(confidence,2)}%'
        
        # Créer un message multipart avec le texte et l'image
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = email_address
        msg['To'] = email_addressre
        
        # Ajouter le texte au message
        msg.attach(MIMEText(f'Une détection avec une confiance supérieure à {round(confidence,2)} a été effectuée.'))
        
        # Convertir l'image en format jpg
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Créer une pièce jointe d'image
        msg.attach(MIMEImage(buffer.tobytes(), name="detection.jpg"))

        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()

        video_attachment = MIMEApplication(video_data, _subtype='mp4')
        video_attachment.add_header('content-disposition', 'attachment', filename='video_captured.mp4')
        msg.attach(video_attachment)


        
        try:

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
            server.quit()

            print('E-mail envoyé avec succès.')
            logging.info("E-mail envoyé avec succès.")

        except Exception as e:

            print('Erreur lors de l\'envoi de l\'e-mail:', str(e))
            logging.error(f"Erreur lors de l\'envoi de l\'e-mail: {str(e)}")



    

    # Running real time 
    cap = cv2.VideoCapture(stream)

    # Crée une instance du modèle YOLO v8 en utilisant le fichier de poids pré-entraîné "best.pt"
    model = YOLO('best.pt')

    # Reading the classes
    classnames = ['fire', 'smoke']


    dernier_envoi_temps = 0
    premier_envoi_temps = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skipper = fps // 2
    skip = 1
    time_skip_alert = 1800
    n = 0
    frame_count = 0
    n_frame = 15 * fps
    informations = []
    frames = []


    logging.info("Démarrage de l'application.")

    while True:
        try:

            if skip % 6 == 0:
                for _ in range(frame_skipper):
                    cap.grab()

            ret, frame = cap.retrieve()

            if not ret:
                logging.warning(f"Reconnecting stream ....")
                cap = cv2.VideoCapture(stream)
                continue

            frames.append(frame)

            frame = cv2.resize(frame, (640, 480))
            result = model(frame, stream=True)

            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 30:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                        informations.append([confidence, frame])

            frame_count += 1
            skip += 1

            cv2.imshow('frame', frame)

            cv2.waitKey(0)


            # Le système d'alerte
            if frame_count >= n_frame:

                if (n_frame >= (450 * fps) and len(informations) == 0 ):
                    send_sms(mean_confidence , False)
                    logging.warning("la derniere detection est probablement fausse.")


                if len(informations) >= (n_frame / 5):
                    
                    frameAlert = None
                    confs = []
                    for ite in informations:
                        confs.append(ite[0])
                    
                    mean_confidence = np.mean(confs)

                    max_confidence = np.max(confs)

                    for item in informations:
                        if item[0] == max_confidence:
                            frameAlert = item[1]
                            break

                    if time.time() - dernier_envoi_temps >= time_skip_alert:

                        if time.time() - premier_envoi_temps >= time_skip_alert:

                            frameAlert_index = frames.index(frameAlert)
                            start_index = max(0, frameAlert_index - 10 * fps)  
                            end_index = min(len(frames), frameAlert_index + 10 * fps + 1)
                            selected_frames = frames[start_index:end_index]

                            video_writer = cv2.VideoWriter('video_captured.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))

                            for frame in selected_frames:
                                video_writer.write(frame)
                            video_writer.release()

                            send_sms(mean_confidence , True)
                            alert_video_path = 'video_captured.mp4' 
                            send_email(frameAlert, max_confidence, alert_video_path)

                            frames.clear()

                            

                            dernier_envoi_temps = time.time()
                            if premier_envoi_temps == 0:
                                premier_envoi_temps = time.time()
                            
                            n_frame = 450 * fps
                            n+=1
                else:
                    n_frame = 15 * fps
                    n = 0


                informations.clear()
                premier_envoi_temps == 0
                frame_count = 0

            if n == 3 :
                    time_skip_alert = 10800
                    n_frame = 2700 * fps
                    n = 0
            
        except Exception as e:
            logging.error(f"Une erreur s'est produite dans la boucle principale : {str(e)}")

    # Libérer la capture vidéo 
    cap.release()
