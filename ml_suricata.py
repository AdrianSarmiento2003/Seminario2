import json
import time
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Program Files\Suricata")  
EVE_FILE = BASE_DIR / "log" / "eve.json"      
MODEL_FILE = Path(r"C:\Program Files\Suricata\rf_pipeline_cicids.joblib") 
ALERT_LOG = BASE_DIR / "log" / "ml_alerts.json"
THRESHOLD = 0.7  

# Cargar modelo
model = joblib.load(MODEL_FILE)
print(f"[INFO] Modelo cargado: {MODEL_FILE}")

# Features esperadas (mismas 43 usadas en entrenamiento)
FEATURES = [
    "Destination Port","Flow Duration",
    "Total Fwd Packets","Total Length of Fwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min",
    "Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min",
    "Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","PSH Flag Count","ACK Flag Count",
    "Average Packet Size","Active Mean","Active Max","Active Min",
    "Idle Mean","Idle Max","Idle Min"
]

# Mapeo Suricata -> CICIDS features 
#Fue realizado la funcion def extract_features(ev):
# El presente código fue generado usando la IA ChatGPT, el prompt fue el siguiente "integrar rf_pipeline_cicids.joblib a Suricata".


def extract_features(ev):
    flow = ev.get("flow", {})
    feats = {}

    # Destination Port
    feats["Destination Port"] = ev.get("dest_port", 0)

    # Flow basics
    feats["Flow Duration"] = flow.get("duration", 0)
    feats["Total Fwd Packets"] = flow.get("pkts_toserver", 0)

    # Ejemplo: cálculo simple de Bwd Packet Length Mean
    feats["Bwd Packet Length Mean"] = flow.get("bytes_toclient", 0) / max(flow.get("pkts_toclient", 1), 1)

    # TCP Flags
    feats["FIN Flag Count"] = 1 if "F" in ev.get("tcp_flags", "") else 0
    feats["PSH Flag Count"] = 1 if "P" in ev.get("tcp_flags", "") else 0
    feats["ACK Flag Count"] = 1 if "A" in ev.get("tcp_flags", "") else 0

    # Completar con 0 los que no se puedan calcular
    for f in FEATURES:
        feats.setdefault(f, 0)

    # Devolver en el mismo orden que el modelo espera
    return [feats[f] for f in FEATURES]

# Tail file generator
def follow(path):
    with open(path, "r") as f:
        f.seek(0, 2)  
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# Guardar alerta
def log_alert(event, score):
    alert = {
        "timestamp": event.get("timestamp"),
        "src_ip": event.get("src_ip"),
        "dest_ip": event.get("dest_ip"),
        "proto": event.get("proto"),
        "ml_score": score,
        "ml_alert": "Suspicious Flow (ML)"
    }
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")
    print("[ALERTA-ML]", alert)

def main():
    print("[INFO] Iniciando extensión ML para Suricata (Windows)...")
    print(f"[INFO] Escuchando: {EVE_FILE}")
    for ev in follow(EVE_FILE):
        if ev.get("event_type") != "flow":
            continue
        feats = extract_features(ev)
        try:
            prob = model.predict_proba([feats])[0][1]
        except AttributeError:
            prob = model.predict([feats])[0]
        if prob >= THRESHOLD:
            log_alert(ev, float(prob))

if __name__ == "__main__":
    main()
