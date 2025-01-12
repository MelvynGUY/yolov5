import cv2
import torch
import numpy as np

# Charger le modèle YOLOv5 pré-entraîné
model_path = "yolov5s.pt"  # Remplacez par le chemin vers votre modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Charger la vidéo
video_path = "test5.mp4"  # Remplacez par votre fichier vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur: Impossible de lire la vidéo.")
    exit()

# Liste pour stocker les régions des personnes détectées et leurs ID
person_regions = {}  # ID: [x_min, y_min, x_max, y_max, vx, vy, frames_since_last_seen]
person_count = 0
total_person_count = 0  # Compteur total de personnes détectées
MAX_DISTANCE = 200  # Distance maximale pour associer deux détections (en pixels)
FRAMES_BEFORE_REMOVAL = 30  # Nombre de frames avant de supprimer une personne "perdue"

def calculate_centroid(x_min, y_min, x_max, y_max):
    """Calculer le centroïde d'une détection (le centre de la boîte)."""
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def predict_position(region):
    """Prédit la nouvelle position d'une personne en fonction de sa vitesse."""
    x_min, y_min, x_max, y_max, vx, vy, _ = region
    return [
        x_min + vx, y_min + vy,
        x_max + vx, y_max + vy
    ]

def associate_detections_to_tracks(detections, tracks, max_distance):
    """
    Associe les détections aux pistes existantes en fonction de la distance.
    """
    unmatched_detections = []
    unmatched_tracks = list(tracks.keys())
    matches = []

    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, cls = detection
        if confidence < 0.5 or cls != 0:  # Classe 0 = personne
            continue
        detection_centroid = calculate_centroid(x_min, y_min, x_max, y_max)

        best_match = None
        min_distance = max_distance

        for track_id in unmatched_tracks:
            track_data = tracks[track_id]
            predicted_position = predict_position(track_data)
            track_centroid = calculate_centroid(*predicted_position[:4])
            distance = np.sqrt((detection_centroid[0] - track_centroid[0])**2 +
                               (detection_centroid[1] - track_centroid[1])**2)
            if distance < min_distance:
                best_match = track_id
                min_distance = distance

        if best_match is not None:
            matches.append((best_match, detection))
            unmatched_tracks.remove(best_match)
        else:
            unmatched_detections.append(detection)

    return matches, unmatched_detections, unmatched_tracks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer YOLO pour détecter des personnes
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Format : [x_min, y_min, x_max, y_max, confidence, class]

    # Associer détections et pistes
    matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(detections, person_regions, MAX_DISTANCE)

    # Mettre à jour les pistes existantes
    for track_id, detection in matches:
        x_min, y_min, x_max, y_max, _, _ = detection
        old_region = person_regions[track_id]
        old_centroid = calculate_centroid(*old_region[:4])
        new_centroid = calculate_centroid(x_min, y_min, x_max, y_max)
        vx, vy = new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1]
        person_regions[track_id] = [int(x_min), int(y_min), int(x_max), int(y_max), vx, vy, 0]

    # Ajouter de nouvelles pistes pour les détections non associées
    for detection in unmatched_detections:
        x_min, y_min, x_max, y_max, _, _ = detection
        person_regions[person_count] = [int(x_min), int(y_min), int(x_max), int(y_max), 0, 0, 0]
        person_count += 1
        total_person_count += 1  # Incrémenter le compteur total

    # Incrémenter le compteur de frames pour les pistes non associées
    for track_id in unmatched_tracks:
        person_regions[track_id][6] += 1

    # Supprimer les pistes dépassant la durée de vie
    person_regions = {k: v for k, v in person_regions.items() if v[6] <= FRAMES_BEFORE_REMOVAL}

    # Dessiner les rectangles et les IDs
    for track_id, (x_min, y_min, x_max, y_max, vx, vy, _) in person_regions.items():
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher le nombre de personnes détectées
    cv2.putText(frame, f"Personnes détectées : {len(person_regions)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher la vidéo
    cv2.imshow("Video avec détection", frame)

    # Arrêter sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Afficher le total des personnes détectées
print(f"Nombre total de personnes détectées : {total_person_count}")
