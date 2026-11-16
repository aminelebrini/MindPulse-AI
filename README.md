# MindPulse-AI
Brief du Soutenance Croisée

## Lancer avec Docker

Construire et démarrer Airflow, son interface web et Streamlit :

```bash
docker compose up --build
```

- Airflow : http://localhost:8080
  - utilisateur : `admin`
  - mot de passe : `admin`
- Streamlit : http://localhost:8501

Le DAG `mindpulseai_pipeline` est disponible dans Airflow. Pour arrêter les services :

```bash
docker compose down
```

Le champ `Student_ID` est volontairement exclu des prédictions afin de ne pas utiliser
un identifiant comme variable du modèle.
