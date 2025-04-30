# Previous name QT_Data_17102024, can be found : /home/admin/Documents/GitHub/QT_Robot_2024/src/qtTouch_Data_Reader/QT_Data_17102024.py
# Can be used to visualise the original signals at the same time.
# Can be used for data collection

import spidev
import time
import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard
from datetime import datetime

# Configuration de l'interface SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

# Fonction pour lire les données de l'ADC
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Générer un nom de fichier unique basé sur la date du jour
date_du_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = f'Data_offline_TEST_31_JAN_{date_du_jour}.csv'

# Initialiser un DataFrame vide
df = pd.DataFrame(columns=['Temps', 'Capteur Triboélectrique', 'Capteur de Pression'])

#Variable pour arreter l'acquisition
acquisition_running = True

#Fonction pour arreter l'acquisition 
def on_press(key):
    global acquisition_running
    if key.char == 'q':
        acquisition_running = False
        return False
    
listener = keyboard.Listener(on_press = on_press)
listener.start()

# Lire les données jusqu'à l'appui d'une touche
start_time = time.time()


while acquisition_running:
    timestamp = time.time() - start_time  # Temps écoulé depuis le début
    print("Time : ", timestamp)
    tribo_dataLF = read_adc(0)
    pressure_dataLF = read_adc(1)  
    tribo_dataRF = read_adc(2)
    pressure_dataRF = read_adc(3)
    tribo_dataLA = read_adc(4)
    pressure_dataLA = read_adc(5)  # Commenté  car CH1 n'est pas utilisé
    tribo_dataRA = read_adc(6)
    pressure_dataRA = read_adc(7)  # Commenté car CH1 n'est pas utilisé

    # Ajouter les données au DataFrame
    new_data = pd.DataFrame([[timestamp, tribo_dataLF, pressure_dataLF
    , tribo_dataRF,pressure_dataRF , tribo_dataLA, pressure_dataLA , tribo_dataRA, pressure_dataRA
    ]]
    , columns=['Temps', 'Capteur Triboélectrique LF', 'Capteur de Pression LF'
    , 'Capteur Triboélectrique RF', 'Capteur de Pression RF' , 'Capteur Triboélectrique LA', 'Capteur de Pression LA' , 'Capteur Triboélectrique RA', 'Capteur de Pression RA' 
    ])
    df = pd.concat([df, new_data], ignore_index=True)



    time.sleep(0.05)  # Intervalle de 0.05 seconde entre les lectures Pour changer le sampling rate


#plt.ioff()
#plt.show()

# Sauvegarder le DataFrame dans un fichier CSV
df.to_csv(file_path, index=False)
print(f"Données sauvegardées dans {file_path}")

# Charger les données pour les plots
df = pd.read_csv(file_path)

# Créer le plot pour le capteur triboélectrique
plt.figure(figsize=(18, 7))
# plt.plot(df['Temps'], df['Capteur Triboélectrique'], label='Capteur Triboélectrique')
# plt.xlabel('Temps (s)')
# plt.ylabel('Valeur du Capteur Triboélectrique')
# plt.title('Plot du Capteur Triboélectrique')
# plt.legend()

# Créer le plot pour le capteur de pression
plt.subplot(2,4,1)
plt.plot(df['Temps'], df['Capteur Triboélectrique LF'], label='Capteur Triboélectrique LF')
plt.legend()


plt.subplot(2,4,5)
plt.plot(df['Temps'], df['Capteur de Pression LF'], label='Capteur de Pression LF', color='orange')
plt.xlabel('Temps (s)')
plt.ylabel('Valeur du Capteur de Pression LF')
plt.legend()


plt.subplot(2,4,2)
plt.plot(df['Temps'], df['Capteur Triboélectrique RF'], label='Capteur Triboélectrique RF')
plt.legend()
plt.subplot(2,4,6)

plt.plot(df['Temps'], df['Capteur de Pression RF'], label='Capteur de Pression RF', color='orange')
plt.xlabel('Temps (s)')
plt.ylabel('Valeur du Capteur de Pression RF')
plt.legend()


plt.subplot(2,4,3)
plt.plot(df['Temps'], df['Capteur Triboélectrique LA'], label='Capteur Triboélectrique LA')
plt.legend()
plt.subplot(2,4,7)
plt.plot(df['Temps'], df['Capteur de Pression LA'], label='Capteur de Pression LA', color='orange')
plt.xlabel('Temps (s)')
plt.ylabel('Valeur du Capteur de Pression LA')
plt.legend()

plt.subplot(2,4,4)
plt.plot(df['Temps'], df['Capteur Triboélectrique RA'], label='Capteur Triboélectrique RA')
plt.legend()
plt.subplot(2,4,8)
plt.plot(df['Temps'], df['Capteur de Pression RA'], label='Capteur de Pression RA', color='orange')
plt.xlabel('Temps (s)')
plt.ylabel('Valeur du Capteur de Pression RA')
plt.legend()

plt.tight_layout()

plt.show()


plt.figure(figsize=(18, 7))

plt.plot(df['Temps'], df['Capteur Triboélectrique LF'], label='Capteur Triboélectrique LF')
plt.plot(df['Temps'], df['Capteur de Pression LF'], '--', label='Capteur Pression LF')

plt.plot(df['Temps'], df['Capteur Triboélectrique RF'], label='Capteur Triboélectrique RF')
plt.plot(df['Temps'], df['Capteur de Pression RF'], '--', label='Capteur Pression RF')

plt.plot(df['Temps'], df['Capteur Triboélectrique LA'], label='Capteur Triboélectrique LA')
plt.plot(df['Temps'], df['Capteur de Pression LA'], label='Capteur de Pression LA')

plt.plot(df['Temps'], df['Capteur Triboélectrique RA'], label='Capteur Triboélectrique RA')
plt.plot(df['Temps'], df['Capteur de Pression RA'], label='Capteur de Pression RA')
plt.legend()
plt.tight_layout()

plt.show()


# Créer le plot pour le capteur de pression
# plt.figure(figsize=(10, 5))
# plt.plot(df['Temps'], df['Capteur de Pression'], label='Capteur de Pression', color='orange')
# plt.xlabel('Temps (s)')
# plt.ylabel('Valeur du Capteur de Pression')
# plt.title('Plot du Capteur de Pression')
# plt.legend()

# plt.show()
